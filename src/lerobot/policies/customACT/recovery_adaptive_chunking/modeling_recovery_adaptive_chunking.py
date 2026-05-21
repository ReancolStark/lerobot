from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lerobot.policies.customACT.recovery_adaptive_chunking.configuration_recovery_adaptive_chunking import (
    RecoveryAdaptiveChunkingConfig,
)


def _rms(x: Tensor) -> Tensor:
    return torch.sqrt(x.pow(2).mean(dim=-1).clamp_min(0.0))


def _masked_recent_mean(values: Tensor, mask: Tensor, recent_steps: int) -> Tensor:
    if recent_steps > 0:
        values = values[:, -recent_steps:]
        mask = mask[:, -recent_steps:]
    mask_f = mask.to(dtype=values.dtype)
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    return (values * mask_f).sum(dim=1) / denom


def _last_valid(history: Tensor, mask: Tensor) -> Tensor:
    batch_size, _, dim = history.shape
    lengths = mask.long().sum(dim=1)
    gather_idx = (lengths - 1).clamp_min(0).view(batch_size, 1, 1).expand(batch_size, 1, dim)
    values = history.gather(dim=1, index=gather_idx).squeeze(1)
    return torch.where(lengths.unsqueeze(-1) > 0, values, torch.zeros_like(values))


class CausalConv1d(nn.Module):
    """1D causal convolution with explicit left padding."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=dilation,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(x, (self.padding, 0))
        return self.conv1d(x)


class RecoveryAdaptiveChunkingModel(nn.Module):
    """Integrated recovery-token model used by recovery-aware adaptive chunking.

    The module compresses normalized proprioceptive state/action history into a
    small number of ACT encoder tokens and predicts a recovery score. The score
    is trained with ``compute_recovery_score_loss`` and is consumed by the
    adaptive chunk controller during inference.
    """

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        dim_model: int,
        config: RecoveryAdaptiveChunkingConfig,
    ) -> None:
        super().__init__()
        if state_dim <= 0 or action_dim <= 0 or dim_model <= 0:
            raise ValueError("state_dim, action_dim, and dim_model must be positive.")

        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dim_model = dim_model
        self.history_len = config.history_len
        self.num_segments = config.num_segments
        self.use_action_state_error = config.use_action_state_error

        event_hidden_dim = max(1, config.hidden_dim // 2)
        self.action_to_state = nn.Identity() if action_dim == state_dim else nn.Linear(action_dim, state_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(state_dim * 4, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )
        self.conv_layers = nn.ModuleList(
            [
                CausalConv1d(
                    config.hidden_dim,
                    config.hidden_dim,
                    kernel_size=config.conv_kernel_size,
                    dilation=dilation,
                )
                for dilation in config.conv_dilations
            ]
        )
        self.conv_norms = nn.ModuleList([nn.LayerNorm(config.hidden_dim) for _ in config.conv_dilations])
        self.dropout = nn.Dropout(config.dropout)

        self.event_score_mlp = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, event_hidden_dim),
            nn.GELU(),
            nn.Linear(event_hidden_dim, 1),
        )
        self.prior_scale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        self.segment_proj = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, dim_model),
            nn.LayerNorm(dim_model),
            nn.GELU(),
            nn.Linear(dim_model, dim_model),
            nn.LayerNorm(dim_model),
        )
        self.token_pos_embed = nn.Parameter(torch.zeros(config.num_segments, dim_model))
        self.hist_action_head = nn.Linear(dim_model, action_dim)
        self.recovery_score_head = nn.Sequential(
            nn.LayerNorm(dim_model),
            nn.Linear(dim_model, 1),
        )

    def _build_history_features(
        self,
        state_history: Tensor,
        action_history: Tensor,
        current_state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        rel = state_history - current_state.unsqueeze(1)

        vel = torch.zeros_like(state_history)
        vel[:, 1:] = state_history[:, 1:] - state_history[:, :-1]

        acc = torch.zeros_like(state_history)
        acc[:, 1:] = vel[:, 1:] - vel[:, :-1]

        if self.use_action_state_error:
            projected_action = self.action_to_state(action_history)
            exec_error = projected_action - state_history
        else:
            exec_error = torch.zeros_like(state_history)

        hist_feat = torch.cat([rel, vel, acc, exec_error], dim=-1)
        event_prior = vel.norm(dim=-1) + acc.norm(dim=-1) + exec_error.norm(dim=-1)
        return hist_feat, event_prior

    def _segment_boundaries(self, history_len: int) -> list[tuple[int, int]]:
        base = history_len // self.num_segments
        boundaries = []
        start = 0
        for seg_idx in range(self.num_segments):
            end = history_len if seg_idx == self.num_segments - 1 else start + base
            boundaries.append((start, end))
            start = end
        return boundaries

    @staticmethod
    def _masked_mean(x: Tensor, mask: Tensor) -> Tensor:
        mask_f = mask.unsqueeze(-1).to(dtype=x.dtype)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        return (x * mask_f).sum(dim=1) / denom

    @staticmethod
    def _masked_tail(x: Tensor, mask: Tensor) -> Tensor:
        batch_size, _, dim = x.shape
        lengths = mask.long().sum(dim=1)
        gather_idx = (lengths - 1).clamp_min(0).view(batch_size, 1, 1).expand(batch_size, 1, dim)
        tail = x.gather(dim=1, index=gather_idx).squeeze(1)
        return torch.where(lengths.unsqueeze(-1) > 0, tail, torch.zeros_like(tail))

    def forward(
        self,
        *,
        state_history: Tensor,
        action_history: Tensor,
        current_state: Tensor,
        history_mask: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if state_history.ndim != 3:
            raise ValueError(f"state_history must be [B,H,state_dim], got {state_history.shape}")
        if action_history.ndim != 3:
            raise ValueError(f"action_history must be [B,H,action_dim], got {action_history.shape}")
        if current_state.ndim != 2:
            raise ValueError(f"current_state must be [B,state_dim], got {current_state.shape}")
        if history_mask.ndim != 2:
            raise ValueError(f"history_mask must be [B,H], got {history_mask.shape}")

        batch_size, history_len, state_dim = state_history.shape
        if history_len != self.history_len:
            raise ValueError(f"Expected history_len={self.history_len}, got {history_len}")
        if state_dim != self.state_dim:
            raise ValueError(f"Expected state_dim={self.state_dim}, got {state_dim}")
        if action_history.shape[:2] != (batch_size, history_len):
            raise ValueError("action_history must share batch/history dimensions with state_history.")
        if action_history.shape[-1] != self.action_dim:
            raise ValueError(f"Expected action_dim={self.action_dim}, got {action_history.shape[-1]}")
        if current_state.shape != (batch_size, self.state_dim):
            raise ValueError(f"Expected current_state shape {(batch_size, self.state_dim)}, got {current_state.shape}")
        if history_mask.shape != (batch_size, history_len):
            raise ValueError(f"Expected history_mask shape {(batch_size, history_len)}, got {history_mask.shape}")

        mask = history_mask.to(device=state_history.device, dtype=torch.bool)
        hist_feat, event_prior = self._build_history_features(
            state_history,
            action_history,
            current_state,
        )
        hist_feat = hist_feat * mask.unsqueeze(-1).to(dtype=hist_feat.dtype)

        x = self.input_proj(hist_feat)
        for conv, norm in zip(self.conv_layers, self.conv_norms, strict=True):
            y = conv(x.transpose(1, 2)).transpose(1, 2)
            y = self.dropout(F.gelu(y))
            x = norm(x + y)
            x = x * mask.unsqueeze(-1).to(dtype=x.dtype)

        learned_event_score = self.event_score_mlp(x).squeeze(-1)
        event_scores = learned_event_score + self.prior_scale * event_prior
        event_scores = event_scores.masked_fill(~mask, -1e4)

        segment_tokens = []
        for start, end in self._segment_boundaries(history_len):
            seg_x = x[:, start:end]
            seg_mask = mask[:, start:end]
            seg_scores = event_scores[:, start:end]

            trend_feat = self._masked_mean(seg_x, seg_mask)
            tail_feat = self._masked_tail(seg_x, seg_mask)
            seg_weight = torch.softmax(seg_scores, dim=-1) * seg_mask.to(dtype=seg_x.dtype)
            seg_weight = seg_weight / seg_weight.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            event_feat = torch.sum(seg_x * seg_weight.unsqueeze(-1), dim=1)
            segment_tokens.append(torch.cat([trend_feat, tail_feat, event_feat], dim=-1))

        recovery_tokens = self.segment_proj(torch.stack(segment_tokens, dim=1))
        token_summary = recovery_tokens.mean(dim=1)
        aux_outputs = {
            "event_scores": event_scores,
            "event_prior": event_prior,
            "hist_action_pred": self.hist_action_head(token_summary),
            "recovery_score": torch.sigmoid(self.recovery_score_head(token_summary)),
        }
        return recovery_tokens, aux_outputs


def compute_recovery_score_target(
    *,
    state_history: Tensor,
    action_history: Tensor,
    future_actions: Tensor,
    history_mask: Tensor,
    config: RecoveryAdaptiveChunkingConfig,
    action_is_pad: Tensor | None = None,
) -> dict[str, Tensor]:
    """Create a no-label recovery score target from ACT training data."""
    if state_history.ndim != 3:
        raise ValueError(f"state_history must be [B,H,state_dim], got {state_history.shape}")
    if action_history.ndim != 3:
        raise ValueError(f"action_history must be [B,H,action_dim], got {action_history.shape}")
    if future_actions.ndim != 3:
        raise ValueError(f"future_actions must be [B,K,action_dim], got {future_actions.shape}")
    if history_mask.ndim != 2:
        raise ValueError(f"history_mask must be [B,H], got {history_mask.shape}")

    mask = history_mask.to(device=state_history.device, dtype=torch.bool)
    velocity = torch.zeros_like(state_history)
    velocity[:, 1:] = state_history[:, 1:] - state_history[:, :-1]
    acceleration = torch.zeros_like(state_history)
    acceleration[:, 1:] = velocity[:, 1:] - velocity[:, :-1]
    state_motion = _rms(velocity) + _rms(acceleration)

    if config.use_action_state_error and action_history.shape[-1] == state_history.shape[-1]:
        exec_error = _rms(action_history - state_history)
    else:
        exec_error = torch.zeros_like(state_motion)

    recent_exec_error = _masked_recent_mean(exec_error, mask, config.recovery_score_recent_steps)
    recent_state_motion = _masked_recent_mean(state_motion, mask, config.recovery_score_recent_steps)

    last_action = _last_valid(action_history, mask)
    first_future_action = future_actions[:, 0]
    future_action_correction = _rms(first_future_action - last_action)

    if future_actions.shape[1] > 1 and config.recovery_future_action_curvature_weight > 0:
        second_future_action = future_actions[:, 1]
        future_curvature = _rms(second_future_action - 2.0 * first_future_action + last_action)
        if action_is_pad is not None and action_is_pad.shape[1] > 1:
            second_valid = (~action_is_pad[:, 1]).to(device=future_actions.device, dtype=future_actions.dtype)
            future_curvature = future_curvature * second_valid
        future_action_correction = (
            future_action_correction
            + config.recovery_future_action_curvature_weight * future_curvature
        )

    if action_is_pad is not None:
        first_valid = (~action_is_pad[:, 0]).to(device=future_actions.device, dtype=future_actions.dtype)
        future_action_correction = future_action_correction * first_valid

    raw_score = (
        config.recovery_exec_error_weight * recent_exec_error
        + config.recovery_state_motion_weight * recent_state_motion
        + config.recovery_future_action_correction_weight * future_action_correction
    )
    target = torch.sigmoid(
        (raw_score - config.recovery_score_target_center)
        / config.recovery_score_target_temperature
    )

    return {
        "target": target.detach(),
        "raw_score": raw_score.detach(),
        "recent_exec_error": recent_exec_error.detach(),
        "recent_state_motion": recent_state_motion.detach(),
        "future_action_correction": future_action_correction.detach(),
    }


def compute_recovery_score_loss(
    *,
    recovery_score: Tensor,
    state_history: Tensor,
    action_history: Tensor,
    future_actions: Tensor,
    history_mask: Tensor,
    config: RecoveryAdaptiveChunkingConfig,
    action_is_pad: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    target_info = compute_recovery_score_target(
        state_history=state_history,
        action_history=action_history,
        future_actions=future_actions,
        history_mask=history_mask,
        config=config,
        action_is_pad=action_is_pad,
    )
    pred = recovery_score.squeeze(-1).clamp(1e-6, 1.0 - 1e-6)
    target = target_info["target"].to(device=pred.device, dtype=pred.dtype)
    bce = F.binary_cross_entropy(pred, target, reduction="none")

    if action_is_pad is not None:
        valid = (~action_is_pad[:, 0]).to(device=pred.device, dtype=pred.dtype)
        loss = (bce * valid).sum() / valid.sum().clamp_min(1.0)
    else:
        loss = bce.mean()

    return loss, target_info


@dataclass
class AdaptiveActionChunkingDecision:
    chunk_size: int
    base_chunk_size: int
    old_action_weight: float
    state_volatility: float
    state_acceleration: float
    action_uncertainty: float
    recovery_score: float | None
    regime: str


class RecoveryAdaptiveChunkingController:
    """Inference controller that consumes recovery score and predicted ACT chunks."""

    def __init__(
        self,
        config: RecoveryAdaptiveChunkingConfig,
        *,
        policy_chunk_size: int,
        policy_n_action_steps: int,
    ) -> None:
        self.config = config
        runtime_max = min(policy_chunk_size, policy_n_action_steps)
        if config.max_chunk_size > 0:
            runtime_max = min(runtime_max, config.max_chunk_size)
        self.max_chunk_size = max(1, runtime_max)
        self.min_chunk_size = min(max(1, config.min_chunk_size), self.max_chunk_size)
        self.state_history = deque([], maxlen=max(2, config.state_history_len))
        self.previous_chunk_size: int | None = None
        self.latest_decision: AdaptiveActionChunkingDecision | None = None
        self.ensembled_actions: Tensor | None = None
        self.last_executed_action: Tensor | None = None
        self.prediction_count = 0
        self.execution_count = 0
        self.current_chunk_size = 0
        self.current_chunk_step = 0
        self.latest_guard_limit: float | None = None
        self.latest_pre_guard_max_step: float | None = None
        self.latest_post_guard_max_step: float | None = None

    def reset(self) -> None:
        self.state_history.clear()
        self.previous_chunk_size = None
        self.latest_decision = None
        self.ensembled_actions = None
        self.last_executed_action = None
        self.prediction_count = 0
        self.execution_count = 0
        self.current_chunk_size = 0
        self.current_chunk_step = 0
        self.latest_guard_limit = None
        self.latest_pre_guard_max_step = None
        self.latest_post_guard_max_step = None

    def observe_state(self, state: Tensor | None) -> None:
        if state is None:
            return
        state = state.detach()
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.shape[0] != 1:
            return
        self.state_history.append(state[0].float().cpu())

    def observe_executed_action(self, action: Tensor) -> None:
        action = action.detach()
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if action.shape[0] != 1:
            return
        self.last_executed_action = action[0].float().cpu()

    def decide(
        self,
        actions: Tensor,
        *,
        recovery_score: float | None = None,
    ) -> AdaptiveActionChunkingDecision:
        max_chunk_size = min(self.max_chunk_size, actions.shape[1])
        min_chunk_size = min(self.min_chunk_size, max_chunk_size)
        base_chunk_size, action_uncertainty = self._base_chunk_size_from_actions(
            actions,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
        )
        state_volatility, state_acceleration = self._state_motion_stats()

        high_recovery = recovery_score is not None and recovery_score >= self.config.recovery_score_high
        high_action_uncertainty = action_uncertainty >= self.config.action_uncertainty_high
        unstable = (
            state_volatility >= self.config.volatility_high
            or state_acceleration >= self.config.acceleration_high
            or high_recovery
            or high_action_uncertainty
        )
        stable = (
            state_volatility <= self.config.volatility_low
            and state_acceleration < self.config.acceleration_high * 0.5
            and action_uncertainty <= self.config.action_uncertainty_low
            and not high_recovery
        )

        if unstable:
            regime = "unstable"
            scaled_chunk_size = round(base_chunk_size * self.config.unstable_chunk_multiplier)
            old_action_weight = self.config.unstable_old_action_weight
        elif stable:
            regime = "stable"
            scaled_chunk_size = round(base_chunk_size * self.config.stable_chunk_multiplier)
            old_action_weight = self.config.stable_old_action_weight
        else:
            regime = "nominal"
            scaled_chunk_size = base_chunk_size
            old_action_weight = self.config.nominal_old_action_weight

        chunk_size = self._smooth_chunk_size(
            self._clip_chunk_size(scaled_chunk_size, min_chunk_size, max_chunk_size),
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
        )
        old_action_weight = float(
            min(max(old_action_weight, self.config.min_old_action_weight), self.config.max_old_action_weight)
        )

        decision = AdaptiveActionChunkingDecision(
            chunk_size=chunk_size,
            base_chunk_size=base_chunk_size,
            old_action_weight=old_action_weight,
            state_volatility=state_volatility,
            state_acceleration=state_acceleration,
            action_uncertainty=action_uncertainty,
            recovery_score=recovery_score,
            regime=regime,
        )
        self.latest_decision = decision
        return decision

    def update_temporal_ensemble(self, actions: Tensor, *, old_action_weight: float) -> Tensor:
        if self.ensembled_actions is None:
            self.ensembled_actions = actions.clone()
        else:
            overlap = min(self.ensembled_actions.shape[1], actions.shape[1])
            old_weight = torch.as_tensor(old_action_weight, dtype=actions.dtype, device=actions.device)
            merged = old_weight * self.ensembled_actions[:, :overlap] + (1.0 - old_weight) * actions[:, :overlap]
            if actions.shape[1] > overlap:
                self.ensembled_actions = torch.cat([merged, actions[:, overlap:]], dim=1)
            else:
                self.ensembled_actions = merged

        action = self.ensembled_actions[:, 0]
        self.ensembled_actions = self.ensembled_actions[:, 1:]
        if self.ensembled_actions.shape[1] == 0:
            self.ensembled_actions = None
        return action

    def smooth_chunk_transition(self, actions: Tensor) -> Tensor:
        """Blend the first few actions of a new queued chunk with the last executed action."""
        if actions.ndim != 3 or actions.shape[1] == 0:
            return actions

        smoothed = actions
        if self.last_executed_action is None:
            return self._apply_action_continuity_guard(smoothed)

        blend_steps = min(max(0, self.config.transition_blend_steps), actions.shape[1])
        old_weight_max = min(max(self.config.transition_blend_old_action_weight, 0.0), 1.0)
        if blend_steps == 0 or old_weight_max <= 0:
            return self._apply_action_continuity_guard(smoothed)

        last_action = self.last_executed_action.to(device=actions.device, dtype=actions.dtype).view(1, -1)
        if last_action.shape[-1] != actions.shape[-1]:
            return self._apply_action_continuity_guard(smoothed)

        smoothed = actions.clone()
        for step in range(blend_steps):
            old_weight = old_weight_max * (1.0 - step / blend_steps)
            smoothed[:, step] = old_weight * last_action + (1.0 - old_weight) * smoothed[:, step]
        return self._apply_action_continuity_guard(smoothed)

    def _apply_action_continuity_guard(self, actions: Tensor) -> Tensor:
        """Limit large action jumps inside the selected execution prefix.

        Adaptive chunking can shorten the task by executing more decisive action
        prefixes, but the touch-task analysis showed that this may slightly raise
        the mean per-frame joint step. This guard keeps the controller's
        efficiency benefit while clipping only deltas that are outliers relative
        to the current predicted chunk.
        """
        if (
            not self.config.action_continuity_guard
            or actions.ndim != 3
            or actions.shape[1] == 0
        ):
            self.latest_guard_limit = None
            self.latest_pre_guard_max_step = None
            self.latest_post_guard_max_step = None
            return actions

        if self.last_executed_action is not None:
            reference = self.last_executed_action.to(device=actions.device, dtype=actions.dtype).view(1, -1)
            if reference.shape[-1] != actions.shape[-1]:
                reference = actions[:, 0].detach()
        else:
            reference = actions[:, 0].detach()
        if reference.shape[0] != actions.shape[0]:
            reference = reference.expand(actions.shape[0], -1)

        sequence = torch.cat([reference.unsqueeze(1), actions], dim=1)
        deltas = sequence[:, 1:] - sequence[:, :-1]
        step_norm = _rms(deltas).detach().flatten()
        if step_norm.numel() == 0:
            self.latest_guard_limit = None
            self.latest_pre_guard_max_step = None
            self.latest_post_guard_max_step = None
            return actions
        self.latest_pre_guard_max_step = float(step_norm.max().item())

        median = step_norm.median()
        mad = (step_norm - median).abs().median()
        limit = median + self.config.action_step_guard_mad_scale * mad
        min_norm = torch.as_tensor(
            self.config.action_step_guard_min_norm,
            dtype=actions.dtype,
            device=actions.device,
        )
        limit = torch.maximum(limit.to(device=actions.device, dtype=actions.dtype), min_norm)
        if float(limit.item()) <= 0.0:
            self.latest_guard_limit = None
            self.latest_post_guard_max_step = self.latest_pre_guard_max_step
            return actions
        self.latest_guard_limit = float(limit.item())

        guarded = actions.clone()
        prev = reference
        eps = torch.finfo(actions.dtype).eps
        for step in range(actions.shape[1]):
            delta = guarded[:, step] - prev
            norm = _rms(delta).unsqueeze(-1).clamp_min(eps)
            scale = torch.clamp(limit / norm, max=1.0)
            guarded[:, step] = prev + delta * scale
            prev = guarded[:, step]
        guarded_sequence = torch.cat([reference.unsqueeze(1), guarded], dim=1)
        guarded_deltas = guarded_sequence[:, 1:] - guarded_sequence[:, :-1]
        self.latest_post_guard_max_step = float(_rms(guarded_deltas).detach().max().item())
        return guarded

    def debug_prediction(
        self,
        *,
        predicted_actions: Tensor,
        executed_actions: Tensor,
        decision: AdaptiveActionChunkingDecision | None,
        source: str,
    ) -> None:
        if not self.config.debug_print_chunks:
            return
        self.prediction_count += 1
        predicted_len = predicted_actions.shape[1]
        executed_len = executed_actions.shape[1]
        self.current_chunk_size = executed_len
        self.current_chunk_step = 0

        every = max(1, self.config.debug_print_every)
        if (self.prediction_count - 1) % every != 0:
            return

        if decision is None:
            metrics = "aac=off"
        else:
            recovery = "none" if decision.recovery_score is None else f"{decision.recovery_score:.4f}"
            metrics = (
                f"regime={decision.regime} base_k={decision.base_chunk_size} "
                f"k_final={decision.chunk_size} old_w={decision.old_action_weight:.3f} "
                f"state_v={decision.state_volatility:.4f} state_a={decision.state_acceleration:.4f} "
                f"act_u={decision.action_uncertainty:.4f} recovery={recovery}"
            )
            if self.latest_guard_limit is not None:
                metrics += (
                    f" guard_limit={self.latest_guard_limit:.4f} "
                    f"step_max={self.latest_pre_guard_max_step:.4f}->{self.latest_post_guard_max_step:.4f}"
                )

        print(
            f"[RAC][predict #{self.prediction_count}][{source}] "
            f"predicted_chunk={predicted_len} executed_chunk={executed_len} {metrics}",
            flush=True,
        )
        print(
            f"[RAC][predict #{self.prediction_count}] predicted_preview={self._preview_actions(predicted_actions)}",
            flush=True,
        )
        print(
            f"[RAC][predict #{self.prediction_count}] executed_preview={self._preview_actions(executed_actions)}",
            flush=True,
        )

    def debug_execution(self, *, action: Tensor, remaining_actions: int, source: str) -> None:
        if not self.config.debug_print_chunks:
            return
        self.execution_count += 1
        self.current_chunk_step += 1
        print(
            f"[RAC][execute #{self.execution_count}][{source}] "
            f"chunk_step={self.current_chunk_step}/{self.current_chunk_size} "
            f"remaining={remaining_actions} action={self._preview_action(action)}",
            flush=True,
        )

    def _state_motion_stats(self) -> tuple[float, float]:
        if len(self.state_history) < 2:
            return 0.0, 0.0
        states = torch.stack(list(self.state_history), dim=0)
        velocity = states[1:] - states[:-1]
        volatility = torch.sqrt(velocity.pow(2).mean(dim=-1).clamp_min(0.0)).mean()
        if velocity.shape[0] < 2:
            return float(volatility.item()), 0.0
        acceleration = velocity[1:] - velocity[:-1]
        acceleration_score = torch.sqrt(acceleration.pow(2).mean(dim=-1).clamp_min(0.0)).mean()
        return float(volatility.item()), float(acceleration_score.item())

    def _base_chunk_size_from_actions(
        self,
        actions: Tensor,
        *,
        min_chunk_size: int,
        max_chunk_size: int,
    ) -> tuple[int, float]:
        if actions.ndim != 3:
            raise ValueError(f"actions must have shape [B,K,action_dim], got {actions.shape}")
        if actions.shape[1] <= min_chunk_size:
            return actions.shape[1], 0.0

        action_seq = actions[0, :max_chunk_size].detach().float()
        velocity = action_seq[1:] - action_seq[:-1]
        step_score = torch.sqrt(velocity.pow(2).mean(dim=-1).clamp_min(0.0))

        if action_seq.shape[0] >= 3 and self.config.action_curvature_weight > 0:
            curvature = action_seq[2:] - 2 * action_seq[1:-1] + action_seq[:-2]
            curvature_score = torch.sqrt(curvature.pow(2).mean(dim=-1).clamp_min(0.0))
            step_score = step_score.clone()
            step_score[1:] = step_score[1:] + self.config.action_curvature_weight * curvature_score

        uncertainty = float(step_score.mean().item()) if step_score.numel() else 0.0
        if step_score.numel() < 2 or torch.all(step_score < 1e-6):
            return max_chunk_size, uncertainty

        score_change = (step_score[1:] - step_score[:-1]).abs()
        candidate_positions = torch.arange(2, max_chunk_size, device=score_change.device)
        valid = (candidate_positions >= min_chunk_size) & (candidate_positions <= max_chunk_size)
        if not valid.any():
            return max_chunk_size, uncertainty

        valid_changes = score_change[valid]
        valid_positions = candidate_positions[valid]
        base_chunk_size = int(valid_positions[torch.argmax(valid_changes)].item())
        return self._clip_chunk_size(base_chunk_size, min_chunk_size, max_chunk_size), uncertainty

    def _smooth_chunk_size(self, chunk_size: int, *, min_chunk_size: int, max_chunk_size: int) -> int:
        if self.previous_chunk_size is None:
            self.previous_chunk_size = chunk_size
            return chunk_size

        smoothing = min(max(self.config.chunk_smoothing, 0.0), 1.0)
        smoothed = round((1.0 - smoothing) * self.previous_chunk_size + smoothing * chunk_size)
        if self.config.max_chunk_delta > 0:
            max_delta = self.config.max_chunk_delta
            smoothed = min(max(smoothed, self.previous_chunk_size - max_delta), self.previous_chunk_size + max_delta)
        smoothed = self._clip_chunk_size(smoothed, min_chunk_size, max_chunk_size)
        self.previous_chunk_size = smoothed
        return smoothed

    @staticmethod
    def _clip_chunk_size(chunk_size: int, min_chunk_size: int, max_chunk_size: int) -> int:
        return int(min(max(chunk_size, min_chunk_size), max_chunk_size))

    def _preview_actions(self, actions: Tensor) -> list[list[float]]:
        num_actions = max(0, self.config.debug_print_num_actions)
        action_dims = max(0, self.config.debug_print_action_dims)
        if num_actions == 0 or action_dims == 0:
            return []
        preview = actions[0, :num_actions, :action_dims].detach().float().cpu()
        return [[round(float(v), 4) for v in row] for row in preview]

    def _preview_action(self, action: Tensor) -> list[float]:
        action_dims = max(0, self.config.debug_print_action_dims)
        if action_dims == 0:
            return []
        action = action.detach().float().cpu()
        if action.ndim == 2:
            action = action[0]
        return [round(float(v), 4) for v in action[:action_dims]]


__all__ = [
    "AdaptiveActionChunkingDecision",
    "CausalConv1d",
    "RecoveryAdaptiveChunkingController",
    "RecoveryAdaptiveChunkingModel",
    "compute_recovery_score_loss",
    "compute_recovery_score_target",
]
