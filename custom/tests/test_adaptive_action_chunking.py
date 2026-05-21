from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import torch


repo_root = Path(__file__).resolve().parents[2]


def _load_adaptive_action_chunking_classes():
    module_dir = (
        repo_root
        / "src"
        / "lerobot"
        / "policies"
        / "customACT"
        / "recovery_adaptive_chunking"
    )

    for name in [
        "lerobot",
        "lerobot.policies",
        "lerobot.policies.customACT",
        "lerobot.policies.customACT.recovery_adaptive_chunking",
    ]:
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package

    config_name = (
        "lerobot.policies.customACT.recovery_adaptive_chunking."
        "configuration_recovery_adaptive_chunking"
    )
    config_spec = importlib.util.spec_from_file_location(
        config_name,
        module_dir / "configuration_recovery_adaptive_chunking.py",
    )
    config_module = importlib.util.module_from_spec(config_spec)
    assert config_spec is not None and config_spec.loader is not None
    sys.modules[config_spec.name] = config_module
    config_spec.loader.exec_module(config_module)

    modeling_name = (
        "lerobot.policies.customACT.recovery_adaptive_chunking."
        "modeling_recovery_adaptive_chunking"
    )
    modeling_spec = importlib.util.spec_from_file_location(
        modeling_name,
        module_dir / "modeling_recovery_adaptive_chunking.py",
    )
    modeling_module = importlib.util.module_from_spec(modeling_spec)
    assert modeling_spec is not None and modeling_spec.loader is not None
    sys.modules[modeling_spec.name] = modeling_module
    modeling_spec.loader.exec_module(modeling_module)

    return (
        config_module.RecoveryAdaptiveChunkingConfig,
        modeling_module.RecoveryAdaptiveChunkingController,
    )


AdaptiveActionChunkingConfig, AdaptiveActionChunkingController = _load_adaptive_action_chunking_classes()


def _actions_with_elbow(chunk_size: int = 32, action_dim: int = 4) -> torch.Tensor:
    actions = torch.zeros(1, chunk_size, action_dim)
    for t in range(1, chunk_size):
        step = 0.01 if t < 12 else 0.18
        actions[:, t] = actions[:, t - 1] + step
    return actions


def test_adaptive_chunking_stretches_stable_history():
    cfg = AdaptiveActionChunkingConfig(
        min_chunk_size=4,
        max_chunk_size=32,
        stable_chunk_multiplier=2.0,
        unstable_chunk_multiplier=0.5,
        chunk_smoothing=1.0,
        volatility_low=0.05,
        volatility_high=0.4,
        acceleration_high=0.4,
        action_uncertainty_low=1.0,
        action_uncertainty_high=2.0,
    )
    controller = AdaptiveActionChunkingController(
        cfg,
        policy_chunk_size=32,
        policy_n_action_steps=32,
    )
    for _ in range(8):
        controller.observe_state(torch.zeros(1, 6))

    decision = controller.decide(_actions_with_elbow())

    assert decision.regime == "stable"
    assert decision.chunk_size >= decision.base_chunk_size
    assert decision.old_action_weight == cfg.stable_old_action_weight


def test_adaptive_chunking_shortens_unstable_history():
    cfg = AdaptiveActionChunkingConfig(
        min_chunk_size=4,
        max_chunk_size=32,
        stable_chunk_multiplier=2.0,
        unstable_chunk_multiplier=0.5,
        chunk_smoothing=1.0,
        volatility_low=0.01,
        volatility_high=0.05,
        acceleration_high=0.05,
        action_uncertainty_low=1.0,
        action_uncertainty_high=2.0,
    )
    controller = AdaptiveActionChunkingController(
        cfg,
        policy_chunk_size=32,
        policy_n_action_steps=32,
    )
    for t in range(8):
        controller.observe_state(torch.full((1, 6), float(t * t)))

    decision = controller.decide(_actions_with_elbow())

    assert decision.regime == "unstable"
    assert decision.chunk_size <= decision.base_chunk_size
    assert decision.old_action_weight == cfg.unstable_old_action_weight


def test_adaptive_temporal_ensemble_returns_next_action():
    cfg = AdaptiveActionChunkingConfig(min_chunk_size=2, max_chunk_size=4)
    controller = AdaptiveActionChunkingController(
        cfg,
        policy_chunk_size=4,
        policy_n_action_steps=4,
    )
    first = torch.zeros(1, 4, 3)
    second = torch.ones(1, 4, 3)

    first_action = controller.update_temporal_ensemble(first, old_action_weight=0.5)
    second_action = controller.update_temporal_ensemble(second, old_action_weight=0.25)

    assert first_action.shape == (1, 3)
    assert second_action.shape == (1, 3)
    assert torch.allclose(second_action, torch.full((1, 3), 0.75))


def test_continuity_guard_clips_outlier_action_jump():
    cfg = AdaptiveActionChunkingConfig(
        min_chunk_size=2,
        max_chunk_size=4,
        transition_blend_steps=0,
        action_continuity_guard=True,
        action_step_guard_mad_scale=0.0,
    )
    controller = AdaptiveActionChunkingController(
        cfg,
        policy_chunk_size=4,
        policy_n_action_steps=4,
    )
    controller.observe_executed_action(torch.zeros(1, 1))

    actions = torch.tensor([[[0.0], [0.1], [4.0], [4.1]]])
    guarded = controller.smooth_chunk_transition(actions)
    first_delta = torch.abs(guarded[:, 0] - torch.zeros(1, 1))
    inner_delta = torch.abs(guarded[:, 1:] - guarded[:, :-1])

    assert torch.max(first_delta) <= 0.1001
    assert torch.max(inner_delta) <= 0.1001
    assert guarded[0, 2, 0] < 1.0


if __name__ == "__main__":
    test_adaptive_chunking_stretches_stable_history()
    test_adaptive_chunking_shortens_unstable_history()
    test_adaptive_temporal_ensemble_returns_next_action()
    test_continuity_guard_clips_outlier_action_jump()
    print("Adaptive action chunking tests passed.")
