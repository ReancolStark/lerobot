from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
        self.prediction_count = 0
        self.execution_count = 0
        self.current_chunk_size = 0
        self.current_chunk_step = 0

    def reset(self) -> None:
        self.previous_replan_score = None
        self.previous_chunk_size = None
        self.latest_decision = None
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
            f"k_range=[{decision.min_chunk_size},{decision.max_chunk_size}]",
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
                    "Enable recovery history token scoring or set `fallback_replan_score`."
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
