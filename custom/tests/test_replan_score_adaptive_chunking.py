from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import types

import torch


repo_root = Path(__file__).resolve().parents[2]


def _load_replan_score_adaptive_classes():
    module_dir = (
        repo_root
        / "src"
        / "lerobot"
        / "policies"
        / "customACT"
        / "replan_score_adaptive_chunking"
    )

    for name in [
        "lerobot",
        "lerobot.policies",
        "lerobot.policies.customACT",
        "lerobot.policies.customACT.replan_score_adaptive_chunking",
    ]:
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package

    config_name = (
        "lerobot.policies.customACT.replan_score_adaptive_chunking."
        "configuration_adaptive_chunking"
    )
    config_spec = importlib.util.spec_from_file_location(
        config_name,
        module_dir / "configuration_adaptive_chunking.py",
    )
    config_module = importlib.util.module_from_spec(config_spec)
    assert config_spec is not None and config_spec.loader is not None
    sys.modules[config_spec.name] = config_module
    config_spec.loader.exec_module(config_module)

    modeling_name = (
        "lerobot.policies.customACT.replan_score_adaptive_chunking."
        "modeling_adaptive_chunking"
    )
    modeling_spec = importlib.util.spec_from_file_location(
        modeling_name,
        module_dir / "modeling_adaptive_chunking.py",
    )
    modeling_module = importlib.util.module_from_spec(modeling_spec)
    assert modeling_spec is not None and modeling_spec.loader is not None
    sys.modules[modeling_spec.name] = modeling_module
    modeling_spec.loader.exec_module(modeling_module)

    return (
        config_module.HistoryTokenAdaptiveChunkingConfig,
        modeling_module.ReplanScoreAdaptiveChunkingController,
    )


HistoryTokenAdaptiveChunkingConfig, ReplanScoreAdaptiveChunkingController = (
    _load_replan_score_adaptive_classes()
)


def _controller(**kwargs) -> ReplanScoreAdaptiveChunkingController:
    cfg = HistoryTokenAdaptiveChunkingConfig(
        min_chunk_size=2,
        max_chunk_size=8,
        debug_print_chunks=False,
        **kwargs,
    )
    return ReplanScoreAdaptiveChunkingController(
        cfg,
        policy_chunk_size=8,
        policy_n_action_steps=8,
    )


def test_replan_score_maps_high_score_to_shorter_chunk():
    controller = _controller(score_smoothing_beta=0.0)

    low = controller.decide(replan_score=0.0, available_actions=8)
    high = controller.decide(replan_score=1.0, available_actions=8)

    assert low.chunk_size == 8
    assert high.chunk_size == 2


def test_boundary_transition_reduces_first_action_jump():
    controller = _controller(
        boundary_transition_steps=3,
        boundary_transition_max_blend=1.0,
        boundary_transition_score_scale=0.01,
    )
    controller.observe_executed_action(torch.zeros(1, 3))
    actions = torch.tensor(
        [
            [
                [1.0, 1.0, 1.0],
                [1.1, 1.1, 1.1],
                [1.2, 1.2, 1.2],
                [1.3, 1.3, 1.3],
            ]
        ]
    )
    decision = controller.decide(replan_score=0.5, available_actions=4)

    smoothed = controller.smooth_chunk_transition(actions, decision=decision)

    before = torch.linalg.vector_norm(actions[:, 0])
    after = torch.linalg.vector_norm(smoothed[:, 0])
    assert after < before
    assert decision.boundary_score is not None
    assert decision.boundary_blend > 0
    assert torch.allclose(smoothed[:, 3], actions[:, 3])


def test_boundary_transition_keeps_actions_without_previous_execution():
    controller = _controller(boundary_transition_steps=3)
    actions = torch.randn(1, 4, 3)

    smoothed = controller.smooth_chunk_transition(actions)

    assert torch.allclose(smoothed, actions)


def test_boundary_transition_can_be_disabled():
    controller = _controller(use_boundary_transition=False)
    controller.observe_executed_action(torch.zeros(1, 3))
    actions = torch.ones(1, 4, 3)

    smoothed = controller.smooth_chunk_transition(actions)

    assert torch.allclose(smoothed, actions)


if __name__ == "__main__":
    test_replan_score_maps_high_score_to_shorter_chunk()
    test_boundary_transition_reduces_first_action_jump()
    test_boundary_transition_keeps_actions_without_previous_execution()
    test_boundary_transition_can_be_disabled()
    print("Replan-score adaptive chunking tests passed.")
