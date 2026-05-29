from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from lerobot.policies.customACT.model_adaptive_chunk.configuration_model_adaptive_chunk import (
    HistoryTokenAdaptiveChunkingConfig,
)


@dataclass
class ReplanScoreAdaptiveChunkingDecision:
    chunk_size: int
    raw_chunk_size: int
    replan_score: float
    smoothed_replan_score: float
    min_chunk_size: int
    max_chunk_size: int
    boundary_score: float | None = None
    boundary_blend: float = 0.0


class ReplanScoreAdaptiveChunkingController:
    """Map a learned replan score to a continuous action chunk execution length."""

    def __init__(
        self,
        config: HistoryTokenAdaptiveChunkingConfig,
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
        self.previous_replan_score: float | None = None
        self.previous_chunk_size: int | None = None
        self.latest_decision: ReplanScoreAdaptiveChunkingDecision | None = None
        self.previous_executed_action: Tensor | None = None
        self.last_executed_action: Tensor | None = None
        self.prediction_count = 0
        self.execution_count = 0
        self.current_chunk_size = 0
        self.current_chunk_step = 0

    def reset(self) -> None:
        self.previous_replan_score = None
        self.previous_chunk_size = None
        self.latest_decision = None
        self.previous_executed_action = None
        self.last_executed_action = None
        self.prediction_count = 0
        self.execution_count = 0
        self.current_chunk_size = 0
        self.current_chunk_step = 0

    def decide(
        self,
        *,
        replan_score: float | None,
        available_actions: int | None = None,
        max_actions_per_chunk: int | None = None,
    ) -> ReplanScoreAdaptiveChunkingDecision:
        score = self._resolve_replan_score(replan_score)
        smoothed_score = self._smooth_replan_score(score)

        max_chunk_size = self.max_chunk_size
        if available_actions is not None:
            max_chunk_size = min(max_chunk_size, available_actions)
        if max_actions_per_chunk is not None:
            max_chunk_size = min(max_chunk_size, max_actions_per_chunk)
        max_chunk_size = max(1, max_chunk_size)
        min_chunk_size = min(self.min_chunk_size, max_chunk_size)

        raw_chunk_size = round(
            min_chunk_size + (1.0 - smoothed_score) * (max_chunk_size - min_chunk_size)
        )
        chunk_size = self._limit_chunk_delta(
            self._clip_chunk_size(raw_chunk_size, min_chunk_size, max_chunk_size),
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
        )

        decision = ReplanScoreAdaptiveChunkingDecision(
            chunk_size=chunk_size,
            raw_chunk_size=raw_chunk_size,
            replan_score=score,
            smoothed_replan_score=smoothed_score,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
        )
        self.latest_decision = decision
        return decision

    def smooth_chunk_transition(
        self,
        actions: Tensor,
        *,
        decision: ReplanScoreAdaptiveChunkingDecision | None = None,
    ) -> Tensor:
        if (
            not self.config.use_boundary_transition
            or self.last_executed_action is None
            or actions.ndim != 3
            or actions.shape[1] == 0
        ):
            if decision is not None:
                decision.boundary_score = None
                decision.boundary_blend = 0.0
            return actions

        max_transition_steps = actions.shape[1] if actions.shape[1] == 1 else actions.shape[1] - 1
        blend_steps = min(max(0, self.config.boundary_transition_steps), max_transition_steps)
        max_blend = min(max(self.config.boundary_transition_max_blend, 0.0), 1.0)
        if blend_steps == 0 or max_blend <= 0:
            if decision is not None:
                decision.boundary_score = None
                decision.boundary_blend = 0.0
            return actions

        last_action = self.last_executed_action.to(device=actions.device, dtype=actions.dtype).view(1, -1)
        if last_action.shape[-1] != actions.shape[-1]:
            if decision is not None:
                decision.boundary_score = None
                decision.boundary_blend = 0.0
            return actions

        prev_action = None
        if self.previous_executed_action is not None:
            prev_action = self.previous_executed_action.to(device=actions.device, dtype=actions.dtype).view(1, -1)
            if prev_action.shape[-1] != actions.shape[-1]:
                prev_action = None

        boundary_score = self._boundary_score(actions, last_action, prev_action)
        blend = self._boundary_blend_weight(boundary_score)
        if decision is not None:
            decision.boundary_score = boundary_score
            decision.boundary_blend = blend
        if blend <= 0:
            return actions

        target_index = min(blend_steps, actions.shape[1] - 1)
        target_action = actions[:, target_index]
        smoothed = actions.clone()
        for step in range(blend_steps):
            progress = float(step + 1) / float(blend_steps + 1)
            eased = self._minimum_jerk(progress)
            bridge = (1.0 - eased) * last_action + eased * target_action
            smoothed[:, step] = blend * bridge + (1.0 - blend) * actions[:, step]
        return smoothed

    def observe_executed_action(self, action: Tensor) -> None:
        action = action.detach()
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if action.shape[0] != 1:
            return
        self.previous_executed_action = self.last_executed_action
        self.last_executed_action = action[0].float().cpu()

    def debug_prediction(
        self,
        *,
        predicted_actions: Any,
        executed_actions: Any,
        decision: ReplanScoreAdaptiveChunkingDecision,
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

        print(
            f"[RSAC][predict #{self.prediction_count}][{source}] "
            f"predicted_chunk={predicted_len} executed_chunk={executed_len} "
            f"raw_k={decision.raw_chunk_size} k_final={decision.chunk_size} "
            f"score={decision.replan_score:.4f} score_smooth={decision.smoothed_replan_score:.4f} "
            f"k_range=[{decision.min_chunk_size},{decision.max_chunk_size}] "
            f"boundary_score={self._format_optional_float(decision.boundary_score)} "
            f"boundary_blend={decision.boundary_blend:.4f}",
            flush=True,
        )
        print(
            f"[RSAC][predict #{self.prediction_count}] predicted_preview="
            f"{self._preview_actions(predicted_actions)}",
            flush=True,
        )
        print(
            f"[RSAC][predict #{self.prediction_count}] executed_preview="
            f"{self._preview_actions(executed_actions)}",
            flush=True,
        )

    def debug_execution(self, *, action: Any, remaining_actions: int, source: str) -> None:
        if not self.config.debug_print_chunks:
            return
        self.execution_count += 1
        self.current_chunk_step += 1
        print(
            f"[RSAC][execute #{self.execution_count}][{source}] "
            f"chunk_step={self.current_chunk_step}/{self.current_chunk_size} "
            f"remaining={remaining_actions} action={self._preview_action(action)}",
            flush=True,
        )

    def _resolve_replan_score(self, replan_score: float | None) -> float:
        if replan_score is None:
            if self.config.fallback_replan_score is None:
                raise ValueError(
                    "Replan-score adaptive chunking is enabled, but no replan score was produced. "
                    "Enable history-token replan scoring or set `fallback_replan_score`."
                )
            replan_score = self.config.fallback_replan_score
        return float(min(max(replan_score, 0.0), 1.0))

    def _smooth_replan_score(self, replan_score: float) -> float:
        beta = min(max(self.config.score_smoothing_beta, 0.0), 1.0)
        if self.previous_replan_score is None:
            smoothed_score = replan_score
        else:
            smoothed_score = beta * self.previous_replan_score + (1.0 - beta) * replan_score
        self.previous_replan_score = float(min(max(smoothed_score, 0.0), 1.0))
        return self.previous_replan_score

    def _limit_chunk_delta(self, chunk_size: int, *, min_chunk_size: int, max_chunk_size: int) -> int:
        if self.previous_chunk_size is None:
            self.previous_chunk_size = chunk_size
            return chunk_size

        if self.config.max_chunk_delta > 0:
            max_delta = self.config.max_chunk_delta
            chunk_size = min(
                max(chunk_size, self.previous_chunk_size - max_delta),
                self.previous_chunk_size + max_delta,
            )
        chunk_size = self._clip_chunk_size(chunk_size, min_chunk_size, max_chunk_size)
        self.previous_chunk_size = chunk_size
        return chunk_size

    @staticmethod
    def _clip_chunk_size(chunk_size: int, min_chunk_size: int, max_chunk_size: int) -> int:
        return int(min(max(chunk_size, min_chunk_size), max_chunk_size))

    def _boundary_score(self, actions: Tensor, last_action: Tensor, prev_action: Tensor | None) -> float:
        curvature_weight = max(0.0, self.config.boundary_transition_curvature_weight)
        first_jump = torch.linalg.vector_norm(actions[:, 0] - last_action, dim=-1)
        score = first_jump
        if prev_action is not None:
            boundary_acc = torch.linalg.vector_norm(actions[:, 0] - 2.0 * last_action + prev_action, dim=-1)
            score = score + curvature_weight * boundary_acc
        if actions.shape[1] > 1:
            early_curvature = torch.linalg.vector_norm(actions[:, 1] - 2.0 * actions[:, 0] + last_action, dim=-1)
            score = score + curvature_weight * early_curvature
        return float(score.detach().mean().item())

    def _boundary_blend_weight(self, boundary_score: float) -> float:
        scale = max(self.config.boundary_transition_score_scale, 1e-6)
        max_blend = min(max(self.config.boundary_transition_max_blend, 0.0), 1.0)
        return max_blend * min(max(boundary_score / scale, 0.0), 1.0)

    @staticmethod
    def _minimum_jerk(progress: float) -> float:
        progress = min(max(progress, 0.0), 1.0)
        return 10.0 * progress**3 - 15.0 * progress**4 + 6.0 * progress**5

    @staticmethod
    def _format_optional_float(value: float | None) -> str:
        if value is None:
            return "n/a"
        return f"{value:.4f}"

    def _preview_actions(self, actions: Any) -> list[list[float]]:
        num_actions = max(0, self.config.debug_print_num_actions)
        action_dims = max(0, self.config.debug_print_action_dims)
        if num_actions == 0 or action_dims == 0:
            return []
        preview = actions[0, :num_actions, :action_dims].detach().float().cpu()
        return [[round(float(v), 4) for v in row] for row in preview]

    def _preview_action(self, action: Any) -> list[float]:
        action_dims = max(0, self.config.debug_print_action_dims)
        if action_dims == 0:
            return []
        action = action.detach().float().cpu()
        if action.ndim == 2:
            action = action[0]
        return [round(float(v), 4) for v in action[:action_dims]]


__all__ = [
    "ReplanScoreAdaptiveChunkingController",
    "ReplanScoreAdaptiveChunkingDecision",
]
