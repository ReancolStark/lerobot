from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch


def _load_recovery_adaptive_classes():
    repo_root = Path(__file__).resolve().parents[2]
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
        pkg = types.ModuleType(name)
        pkg.__path__ = []
        sys.modules[name] = pkg

    config_name = (
        "lerobot.policies.customACT.replan_score_adaptive_chunking."
        "configuration_history_token_replan_score"
    )
    config_spec = importlib.util.spec_from_file_location(
        config_name, module_dir / "configuration_history_token_replan_score.py"
    )
    config_module = importlib.util.module_from_spec(config_spec)
    assert config_spec is not None and config_spec.loader is not None
    sys.modules[config_name] = config_module
    config_spec.loader.exec_module(config_module)

    modeling_name = (
        "lerobot.policies.customACT.replan_score_adaptive_chunking."
        "modeling_history_token_replan_score"
    )
    modeling_spec = importlib.util.spec_from_file_location(
        modeling_name, module_dir / "modeling_history_token_replan_score.py"
    )
    modeling_module = importlib.util.module_from_spec(modeling_spec)
    assert modeling_spec is not None and modeling_spec.loader is not None
    sys.modules[modeling_name] = modeling_module
    modeling_spec.loader.exec_module(modeling_module)
    return config_module.RecoveryAdaptiveChunkingConfig, modeling_module


def test_future_action_correction_increases_recovery_target():
    RecoveryAdaptiveChunkingConfig, modeling = _load_recovery_adaptive_classes()
    cfg = RecoveryAdaptiveChunkingConfig(
        replan_score_target_center=0.5,
        replan_score_target_temperature=0.2,
        replan_score_recent_steps=4,
    )

    B, H, K, D = 2, 8, 4, 3
    state_history = torch.zeros(B, H, D)
    action_history = torch.zeros(B, H, D)
    stable_future = torch.zeros(B, K, D)
    corrected_future = stable_future.clone()
    corrected_future[:, 0] = 4.0
    history_mask = torch.ones(B, H, dtype=torch.bool)
    action_is_pad = torch.zeros(B, K, dtype=torch.bool)

    stable = modeling.compute_recovery_score_target(
        state_history=state_history,
        action_history=action_history,
        future_actions=stable_future,
        history_mask=history_mask,
        action_is_pad=action_is_pad,
        config=cfg,
    )
    corrected = modeling.compute_recovery_score_target(
        state_history=state_history,
        action_history=action_history,
        future_actions=corrected_future,
        history_mask=history_mask,
        action_is_pad=action_is_pad,
        config=cfg,
    )

    assert corrected["future_action_correction"].mean() > stable["future_action_correction"].mean()
    assert corrected["target"].mean() > stable["target"].mean()


def test_recovery_score_loss_is_finite():
    RecoveryAdaptiveChunkingConfig, modeling = _load_recovery_adaptive_classes()
    cfg = RecoveryAdaptiveChunkingConfig(
        replan_score_target_center=0.5,
        replan_score_target_temperature=0.2,
    )

    B, H, K, D = 2, 8, 4, 3
    recovery_score = torch.full((B, 1), 0.5)
    state_history = torch.randn(B, H, D)
    action_history = torch.randn(B, H, D)
    future_actions = torch.randn(B, K, D)
    history_mask = torch.ones(B, H, dtype=torch.bool)
    action_is_pad = torch.zeros(B, K, dtype=torch.bool)

    loss, target_info = modeling.compute_recovery_score_loss(
        recovery_score=recovery_score,
        state_history=state_history,
        action_history=action_history,
        future_actions=future_actions,
        history_mask=history_mask,
        action_is_pad=action_is_pad,
        config=cfg,
    )

    assert torch.isfinite(loss)
    assert target_info["target"].shape == (B,)


def test_integrated_model_and_controller_shapes():
    RecoveryAdaptiveChunkingConfig, modeling = _load_recovery_adaptive_classes()
    cfg = RecoveryAdaptiveChunkingConfig(
        history_len=8,
        num_segments=2,
        hidden_dim=16,
        conv_dilations=[1, 2],
        replan_score_target_center=0.5,
        replan_score_target_temperature=0.2,
        min_chunk_size=2,
        max_chunk_size=8,
    )

    model = modeling.RecoveryAdaptiveChunkingModel(
        state_dim=3,
        action_dim=3,
        dim_model=32,
        config=cfg,
    )
    state_history = torch.randn(2, 8, 3)
    action_history = torch.randn(2, 8, 3)
    current_state = state_history[:, -1]
    history_mask = torch.ones(2, 8, dtype=torch.bool)

    tokens, aux_outputs = model(
        state_history=state_history,
        action_history=action_history,
        current_state=current_state,
        history_mask=history_mask,
    )

    assert tokens.shape == (2, 2, 32)
    assert model.token_pos_embed.shape == (2, 32)
    assert aux_outputs["recovery_score"].shape == (2, 1)

    controller = modeling.RecoveryAdaptiveChunkingController(
        cfg,
        policy_chunk_size=8,
        policy_n_action_steps=8,
    )
    decision = controller.decide(torch.randn(1, 8, 3), recovery_score=0.9)
    assert 2 <= decision.chunk_size <= 8
    assert decision.recovery_score == 0.9


if __name__ == "__main__":
    test_future_action_correction_increases_recovery_target()
    test_recovery_score_loss_is_finite()
    test_integrated_model_and_controller_shapes()
    print("Recovery adaptive chunking tests passed.")
