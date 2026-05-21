from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RecoveryAdaptiveChunkingConfig:
    """Unified config for recovery tokens and recovery-aware adaptive chunking.

    The recovery token is trained from history. The adaptive chunk controller uses
    the trained recovery score at inference time, together with action/state
    volatility, to decide how much of the predicted ACT chunk should be executed.
    """

    enabled: bool = True

    # Module switches.
    use_recovery_token: bool = True
    use_adaptive_action_chunking: bool = True

    # Recovery token encoder.
    history_len: int = 64
    num_segments: int = 4
    hidden_dim: int = 256
    conv_kernel_size: int = 3
    conv_dilations: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    dropout: float = 0.1
    use_action_state_error: bool = True
    token_pos_embed: str = "learned"

    # Recovery token auxiliary losses.
    action_loss_weight: float = 0.05
    event_prior_loss_weight: float = 0.01

    # Recovery score pseudo-label loss.
    recovery_score_loss_weight: float = 0.05
    recovery_score_target_center: float = 1.0
    recovery_score_target_temperature: float = 0.35
    recovery_score_recent_steps: int = 8
    recovery_exec_error_weight: float = 0.45
    recovery_state_motion_weight: float = 0.25
    recovery_future_action_correction_weight: float = 0.30
    recovery_future_action_curvature_weight: float = 0.5

    # Inference-time adaptive action chunking.
    state_history_len: int = 64
    min_chunk_size: int = 8
    max_chunk_size: int = 0
    stable_chunk_multiplier: float = 1.5
    unstable_chunk_multiplier: float = 0.5
    chunk_smoothing: float = 0.35
    max_chunk_delta: int = 16
    volatility_low: float = 0.03
    volatility_high: float = 0.12
    acceleration_high: float = 0.10
    recovery_score_high: float = 0.65
    action_uncertainty_low: float = 0.03
    action_uncertainty_high: float = 0.15
    action_curvature_weight: float = 0.5
    stable_old_action_weight: float = 0.65
    nominal_old_action_weight: float = 0.35
    unstable_old_action_weight: float = 0.05
    min_old_action_weight: float = 0.0
    max_old_action_weight: float = 0.9
    transition_blend_steps: int = 3
    transition_blend_old_action_weight: float = 0.5
    action_continuity_guard: bool = True
    action_step_guard_mad_scale: float = 3.0
    action_step_guard_min_norm: float = 0.0
    debug_print_chunks: bool = True
    debug_print_every: int = 1
    debug_print_num_actions: int = 3
    debug_print_action_dims: int = 6

    @classmethod
    def from_legacy_config(cls, cfg: Any) -> "RecoveryAdaptiveChunkingConfig":
        """Build the unified config from the existing ACTConfig fields.

        This keeps old command lines and checkpoints compatible while exposing a
        single nested config for new experiments.
        """
        unified = deepcopy(getattr(cfg, "recovery_adaptive_chunking", cls()))
        unified.enabled = bool(
            getattr(cfg, "use_recovery_history_token", False)
            or getattr(cfg, "use_adaptive_action_chunking", False)
        )
        unified.use_recovery_token = bool(getattr(cfg, "use_recovery_history_token", False))
        unified.use_adaptive_action_chunking = bool(getattr(cfg, "use_adaptive_action_chunking", False))

        unified.history_len = int(getattr(cfg, "history_len", unified.history_len))
        unified.num_segments = int(getattr(cfg, "history_num_segments", unified.num_segments))
        unified.hidden_dim = int(getattr(cfg, "history_hidden_dim", unified.hidden_dim))
        unified.conv_kernel_size = int(
            getattr(cfg, "history_conv_kernel_size", unified.conv_kernel_size)
        )
        unified.conv_dilations = list(
            getattr(cfg, "history_conv_dilations", unified.conv_dilations)
        )
        unified.dropout = float(getattr(cfg, "history_dropout", unified.dropout))
        unified.use_action_state_error = bool(
            getattr(cfg, "use_action_state_error", unified.use_action_state_error)
        )
        unified.token_pos_embed = str(
            getattr(cfg, "recovery_token_pos_embed", unified.token_pos_embed)
        )
        unified.action_loss_weight = float(
            getattr(cfg, "history_action_loss_weight", unified.action_loss_weight)
        )
        unified.event_prior_loss_weight = float(
            getattr(cfg, "event_prior_loss_weight", unified.event_prior_loss_weight)
        )
        legacy_aac = getattr(cfg, "adaptive_action_chunking", None)
        if legacy_aac is not None:
            for name in (
                "state_history_len",
                "min_chunk_size",
                "max_chunk_size",
                "stable_chunk_multiplier",
                "unstable_chunk_multiplier",
                "chunk_smoothing",
                "max_chunk_delta",
                "volatility_low",
                "volatility_high",
                "acceleration_high",
                "recovery_score_high",
                "action_uncertainty_low",
                "action_uncertainty_high",
                "action_curvature_weight",
                "stable_old_action_weight",
                "nominal_old_action_weight",
                "unstable_old_action_weight",
                "min_old_action_weight",
                "max_old_action_weight",
                "transition_blend_steps",
                "transition_blend_old_action_weight",
                "action_continuity_guard",
                "action_step_guard_mad_scale",
                "action_step_guard_min_norm",
                "debug_print_chunks",
                "debug_print_every",
                "debug_print_num_actions",
                "debug_print_action_dims",
            ):
                if hasattr(legacy_aac, name):
                    setattr(unified, name, deepcopy(getattr(legacy_aac, name)))
        return unified

    def validate(self) -> None:
        if not self.enabled:
            return

        if self.use_recovery_token:
            if self.history_len <= 0:
                raise ValueError("`recovery_adaptive_chunking.history_len` must be positive.")
            if self.num_segments <= 0:
                raise ValueError("`recovery_adaptive_chunking.num_segments` must be positive.")
            if self.history_len < self.num_segments:
                raise ValueError("`recovery_adaptive_chunking.history_len` must be >= num_segments.")
            if self.hidden_dim <= 0:
                raise ValueError("`recovery_adaptive_chunking.hidden_dim` must be positive.")
            if self.conv_kernel_size <= 0:
                raise ValueError("`recovery_adaptive_chunking.conv_kernel_size` must be positive.")
            if not self.conv_dilations:
                raise ValueError("`recovery_adaptive_chunking.conv_dilations` cannot be empty.")
            if self.token_pos_embed != "learned":
                raise ValueError("Only `recovery_adaptive_chunking.token_pos_embed='learned'` is supported.")
            if self.recovery_score_loss_weight < 0:
                raise ValueError("`recovery_score_loss_weight` cannot be negative.")
            if self.recovery_score_target_temperature <= 0:
                raise ValueError("`recovery_score_target_temperature` must be positive.")
            if self.recovery_score_recent_steps <= 0:
                raise ValueError("`recovery_score_recent_steps` must be positive.")
            if min(
                self.recovery_exec_error_weight,
                self.recovery_state_motion_weight,
                self.recovery_future_action_correction_weight,
                self.recovery_future_action_curvature_weight,
            ) < 0:
                raise ValueError("Recovery score target weights cannot be negative.")

        if self.use_adaptive_action_chunking:
            if self.state_history_len < 2:
                raise ValueError("`adaptive_action_chunking.state_history_len` must be >= 2.")
            if self.min_chunk_size <= 0:
                raise ValueError("`adaptive_action_chunking.min_chunk_size` must be positive.")
            if self.max_chunk_size < 0:
                raise ValueError("`adaptive_action_chunking.max_chunk_size` cannot be negative.")
            if self.max_chunk_size and self.max_chunk_size < self.min_chunk_size:
                raise ValueError("`adaptive_action_chunking.max_chunk_size` must be >= min_chunk_size.")
            if self.stable_chunk_multiplier <= 0 or self.unstable_chunk_multiplier <= 0:
                raise ValueError("Adaptive chunk multipliers must be positive.")
            if self.volatility_low < 0 or self.volatility_high < 0 or self.acceleration_high < 0:
                raise ValueError("Adaptive chunk motion thresholds cannot be negative.")
            if self.volatility_low > self.volatility_high:
                raise ValueError("`adaptive_action_chunking.volatility_low` must be <= volatility_high.")
            if self.action_uncertainty_low < 0 or self.action_uncertainty_high < 0:
                raise ValueError("Adaptive chunk action uncertainty thresholds cannot be negative.")
            if self.action_uncertainty_low > self.action_uncertainty_high:
                raise ValueError("`adaptive_action_chunking.action_uncertainty_low` must be <= high.")
            if self.transition_blend_steps < 0:
                raise ValueError("`adaptive_action_chunking.transition_blend_steps` cannot be negative.")
            if not 0 <= self.transition_blend_old_action_weight <= 1:
                raise ValueError(
                    "`adaptive_action_chunking.transition_blend_old_action_weight` must be between 0 and 1."
                )
            if self.action_step_guard_mad_scale < 0:
                raise ValueError("`adaptive_action_chunking.action_step_guard_mad_scale` cannot be negative.")
            if self.action_step_guard_min_norm < 0:
                raise ValueError("`adaptive_action_chunking.action_step_guard_min_norm` cannot be negative.")
            if self.debug_print_every <= 0:
                raise ValueError("`adaptive_action_chunking.debug_print_every` must be positive.")
            if self.debug_print_num_actions < 0 or self.debug_print_action_dims < 0:
                raise ValueError("Adaptive chunk debug preview sizes cannot be negative.")
