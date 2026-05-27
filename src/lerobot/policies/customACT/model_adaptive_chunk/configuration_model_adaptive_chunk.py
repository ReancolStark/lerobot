from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HistoryTokenAdaptiveChunkingConfig:
    """Config for choosing the adaptive chunk execution strategy.

    This is inference-only. It does not change the ACT action head or the fixed
    predicted chunk size.

    Modes:
        replan_score: map the learned replan score continuously to chunk length.
        three_regime: use the previous stable/nominal/unstable controller.
    """

    mode: str = "replan_score"
    min_chunk_size: int = 8
    max_chunk_size: int = 0  # 0 means use the ACT runtime chunk upper bound.
    score_smoothing_beta: float = 0.8
    max_chunk_delta: int = 16
    fallback_replan_score: float | None = None
    debug_print_chunks: bool = True
    debug_print_every: int = 1
    debug_print_num_actions: int = 3
    debug_print_action_dims: int = 6

    def validate(self) -> None:
        if self.mode not in {"replan_score", "three_regime"}:
            raise ValueError(
                "`history_token_adaptive_chunking.mode` must be 'replan_score' or 'three_regime'."
            )
        if self.min_chunk_size <= 0:
            raise ValueError("`history_token_adaptive_chunking.min_chunk_size` must be positive.")
        if self.max_chunk_size < 0:
            raise ValueError("`history_token_adaptive_chunking.max_chunk_size` cannot be negative.")
        if self.max_chunk_size and self.max_chunk_size < self.min_chunk_size:
            raise ValueError(
                "`history_token_adaptive_chunking.max_chunk_size` must be >= min_chunk_size when set."
            )
        if not 0.0 <= self.score_smoothing_beta <= 1.0:
            raise ValueError("`history_token_adaptive_chunking.score_smoothing_beta` must be in [0, 1].")
        if self.max_chunk_delta < 0:
            raise ValueError("`history_token_adaptive_chunking.max_chunk_delta` cannot be negative.")
        if self.fallback_replan_score is not None and not 0.0 <= self.fallback_replan_score <= 1.0:
            raise ValueError("`history_token_adaptive_chunking.fallback_replan_score` must be in [0, 1].")
        if self.debug_print_every <= 0:
            raise ValueError("`history_token_adaptive_chunking.debug_print_every` must be positive.")
        if self.debug_print_num_actions < 0 or self.debug_print_action_dims < 0:
            raise ValueError("Replan-score adaptive chunk debug preview sizes cannot be negative.")


ReplanScoreAdaptiveChunkingConfig = HistoryTokenAdaptiveChunkingConfig
