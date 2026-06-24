from types import SimpleNamespace

import pytest
import torch

from lerobot.policies.customACT.modeling_customACT import ACTPolicy
from lerobot.policies.customACT.mask_weight.configuration_mask_weight import MaskWeightConfig
from lerobot.policies.customACT.mask_weight.mask_weight import (
    MaskGuidedVisualAdapter,
    make_mask_guided_background_augmentation,
    yolo_results_to_object_weight_inputs,
)


def test_mask_weight_config_v4_direct_defaults():
    config = MaskWeightConfig()

    assert config.use_residual_gate is False
    assert config.use_target_tokens is True
    assert config.use_mask_geometry_token is True
    assert config.num_target_tokens == 2
    assert config.gate_init == pytest.approx(0.2)
    assert config.region_attention_gate_init == pytest.approx(0.2)
    assert config.region_attention_context_weight == pytest.approx(0.3)
    assert config.region_attention_background_weight == pytest.approx(0.0)
    assert config.reliability_floor == pytest.approx(0.6)
    assert config.mask_noise_p == pytest.approx(0.0)
    assert config.mask_noise_dropout_p == pytest.approx(0.0)
    assert config.background_feature_consistency_loss_weight == pytest.approx(0.03)
    assert config.background_token_consistency_loss_weight == pytest.approx(0.03)
    assert config.use_target_token_attention is True
    assert config.target_token_attention_heads == 4
    assert config.target_token_attention_gate_init == pytest.approx(0.12)
    assert config.target_token_attention_gate_max == pytest.approx(0.2)
    assert config.target_token_delta_ratio_limit == pytest.approx(0.45)
    assert config.target_token_mask_bias_scale == pytest.approx(1.2)
    assert config.target_object_perceiver_layers == 2
    assert config.target_object_perceiver_ffn_dim == 1024
    assert config.target_object_perceiver_dropout == pytest.approx(0.0)
    assert config.mask_geometry_token_gate_init == pytest.approx(0.2)
    assert config.target_background_contrastive_loss_weight == pytest.approx(0.01)
    assert config.target_background_contrastive_margin == pytest.approx(0.2)
    assert config.target_background_contrastive_use_object_features is False
    assert config.use_object_weighted_mask is True
    assert config.max_object_weight_instances == 8
    assert config.object_weight_embed_dim == 64
    assert config.object_weight_hidden_dim == 128
    assert config.object_weight_relation_layers == 1
    assert config.object_weight_attention_heads == 4
    assert config.object_weight_init == pytest.approx(0.8)
    assert config.object_weight_union_floor == pytest.approx(0.7)
    assert config.object_token_mask_mix == pytest.approx(0.2)
    assert config.num_yolo_classes is None


def test_mask_guided_visual_adapter_shapes_and_gradient():
    torch.manual_seed(0)
    config = MaskWeightConfig(
        adapter_hidden_dim=4,
        mask_dropout_p=0.0,
        mask_noise_p=0.0,
        use_residual_gate=True,
        use_target_tokens=False,
    )
    adapter = MaskGuidedVisualAdapter(dim_model=8, config=config)

    features = torch.randn(2, 8, 5, 6, requires_grad=True)
    mask = torch.zeros(2, 1, 5, 6)
    mask[:, :, 1:4, 2:5] = 0.8

    guided_features, target_tokens, target_pos_embed, mask_geometry_token, mask_geometry_pos_embed = adapter(
        features,
        mask,
    )

    assert guided_features.shape == features.shape
    assert target_tokens is None
    assert target_pos_embed is None
    assert mask_geometry_token.shape == (1, 2, 8)
    assert mask_geometry_pos_embed.shape == (1, 1, 8)
    assert adapter.latest_debug["target_background_rms_ratio_before"] > 0.0
    assert adapter.latest_debug["target_background_rms_ratio_after"] > 0.0
    assert adapter.latest_debug["target_background_rms_ratio_gain"] > 0.0
    assert adapter.latest_debug["target_delta_ratio"] >= 0.0
    assert adapter.latest_debug["background_delta_ratio"] >= 0.0
    assert adapter.latest_debug["reliability_mean"] > 0.0
    assert adapter.latest_debug["region_attention_delta_ratio"] >= 0.0
    assert adapter.latest_debug["region_attention_gate"] == pytest.approx(config.region_attention_gate_init)

    guided_features.mean().backward()
    assert adapter.gate_scale.grad is not None
    assert adapter.region_attention_scale.grad is not None


def test_mask_guided_visual_adapter_target_tokens():
    torch.manual_seed(0)
    config = MaskWeightConfig(
        adapter_hidden_dim=4,
        mask_dropout_p=0.0,
        mask_noise_p=0.0,
        use_target_tokens=True,
        num_target_tokens=3,
    )
    adapter = MaskGuidedVisualAdapter(dim_model=8, config=config)

    features = torch.randn(2, 8, 5, 6)
    mask = torch.zeros(2, 1, 5, 6)
    mask[:, :, 1:4, 2:5] = 1.0

    guided_features, target_tokens, target_pos_embed, mask_geometry_token, mask_geometry_pos_embed = adapter(
        features,
        mask,
    )

    assert guided_features.shape == features.shape
    assert target_tokens.shape == (3, 2, 8)
    assert target_pos_embed.shape == (3, 1, 8)
    assert mask_geometry_token.shape == (1, 2, 8)
    assert mask_geometry_pos_embed.shape == (1, 1, 8)
    assert adapter.latest_consistency["target_feature"].shape == (2, 8)
    assert adapter.latest_consistency["background_feature"].shape == (2, 8)
    assert adapter.latest_consistency["target_tokens"].shape == (3, 2, 8)
    assert adapter.latest_debug["target_token_attention_delta_ratio"] >= 0.0
    assert adapter.latest_debug["object_perceiver_delta_ratio"] >= 0.0
    assert adapter.latest_debug["object_perceiver_layers"] == pytest.approx(2.0)
    assert adapter.latest_debug["target_token_attention_gate"] == pytest.approx(
        config.target_token_attention_gate_init
    )
    assert adapter.latest_debug["object_perceiver_gate"] == pytest.approx(
        config.target_token_attention_gate_init
    )
    assert adapter.latest_debug["target_token_geometry_norm"] == pytest.approx(0.0)
    assert adapter.latest_debug["mask_geometry_token_norm"] >= 0.0
    assert adapter.latest_debug["mask_geometry_token_gate"] == pytest.approx(
        config.mask_geometry_token_gate_init
    )
    assert 0.0 <= adapter.latest_debug["mask_geometry_area"] <= 1.0
    assert 0.0 <= adapter.latest_debug["mask_geometry_confidence"] <= 1.0
    assert 0.0 <= adapter.latest_debug["mask_geometry_reliability"] <= 1.0
    assert 0.0 <= adapter.latest_debug["target_token_target_attn_mean"] <= 1.0
    assert 0.0 <= adapter.latest_debug["target_token_context_attn_mean"] <= 1.0

    (target_tokens.mean() + mask_geometry_token.mean()).backward()
    assert adapter.mask_geometry_token_proj[-1].weight.grad is not None
    assert adapter.mask_geometry_token_scale.grad is not None
    assert adapter.target_token_attention_scale.grad is not None
    assert adapter.target_object_perceiver[0].cross_attn.in_proj_weight.grad is not None


def test_target_token_attention_gate_and_delta_are_bounded():
    torch.manual_seed(0)
    config = MaskWeightConfig(
        adapter_hidden_dim=4,
        mask_dropout_p=0.0,
        mask_noise_p=0.0,
        use_target_tokens=True,
        target_token_attention_gate_init=0.2,
        target_token_attention_gate_max=0.2,
        target_token_delta_ratio_limit=0.05,
    )
    adapter = MaskGuidedVisualAdapter(dim_model=8, config=config)
    with torch.no_grad():
        adapter.target_token_attention_scale.fill_(1.0)

    features = torch.randn(2, 8, 5, 6)
    mask = torch.zeros(2, 1, 5, 6)
    mask[:, :, 1:4, 2:5] = 1.0

    _, target_tokens, _, _, _ = adapter(features, mask)

    assert target_tokens.shape == (2, 2, 8)
    assert adapter.latest_debug["target_token_attention_gate"] == pytest.approx(0.2)
    assert adapter.latest_debug["object_perceiver_gate"] == pytest.approx(0.2)
    assert adapter.latest_debug["target_token_attention_gate_raw"] == pytest.approx(1.0)
    assert adapter.latest_debug["target_token_attention_delta_ratio"] <= 0.05 + 1e-6
    assert adapter.latest_debug["object_perceiver_delta_ratio"] <= 0.05 + 1e-6
    assert 0.0 <= adapter.latest_debug["target_token_delta_limiter_min"] <= 1.0
    assert 0.0 <= adapter.latest_debug["target_token_delta_limiter_mean"] <= 1.0


def test_mask_weight_config_rejects_invalid_background_strength():
    with pytest.raises(ValueError):
        MaskWeightConfig(background_aug_min_strength=0.9, background_aug_max_strength=0.2)
    with pytest.raises(ValueError):
        MaskWeightConfig(
            background_consistency_aug_min_strength=0.9,
            background_consistency_aug_max_strength=0.2,
        )


def test_mask_weight_config_rejects_invalid_v4_params():
    with pytest.raises(ValueError):
        MaskWeightConfig(reliability_min_area=0.8, reliability_max_area=0.2)
    with pytest.raises(ValueError):
        MaskWeightConfig(region_attention_heads=0)
    with pytest.raises(ValueError):
        MaskWeightConfig(mask_noise_jitter_px=-1)
    with pytest.raises(ValueError):
        MaskWeightConfig(target_token_attention_heads=0)
    with pytest.raises(ValueError):
        MaskWeightConfig(target_token_attention_dropout=-0.1)
    with pytest.raises(ValueError):
        MaskWeightConfig(target_token_attention_gate_init=-0.1)
    with pytest.raises(ValueError):
        MaskWeightConfig(target_token_attention_gate_max=-0.1)
    with pytest.raises(ValueError):
        MaskWeightConfig(target_token_attention_gate_init=0.3, target_token_attention_gate_max=0.2)
    with pytest.raises(ValueError):
        MaskWeightConfig(target_token_delta_ratio_limit=-0.1)
    with pytest.raises(ValueError):
        MaskWeightConfig(target_token_mask_bias_scale=-0.1)
    with pytest.raises(ValueError):
        MaskWeightConfig(target_object_perceiver_layers=0)
    with pytest.raises(ValueError):
        MaskWeightConfig(target_object_perceiver_ffn_dim=0)
    with pytest.raises(ValueError):
        MaskWeightConfig(target_object_perceiver_dropout=-0.1)
    with pytest.raises(ValueError):
        MaskWeightConfig(mask_geometry_token_gate_init=-0.1)
    with pytest.raises(ValueError):
        MaskWeightConfig(max_object_weight_instances=-1)
    with pytest.raises(ValueError):
        MaskWeightConfig(object_weight_embed_dim=0)
    with pytest.raises(ValueError):
        MaskWeightConfig(object_weight_hidden_dim=0)
    with pytest.raises(ValueError):
        MaskWeightConfig(object_weight_relation_layers=-1)
    with pytest.raises(ValueError):
        MaskWeightConfig(object_weight_attention_heads=0)
    with pytest.raises(ValueError):
        MaskWeightConfig(object_weight_hidden_dim=10, object_weight_attention_heads=4)
    with pytest.raises(ValueError):
        MaskWeightConfig(object_weight_init=1.0)
    with pytest.raises(ValueError):
        MaskWeightConfig(object_weight_union_floor=-0.1)
    with pytest.raises(ValueError):
        MaskWeightConfig(object_token_mask_mix=-0.1)
    with pytest.raises(ValueError):
        MaskWeightConfig(object_token_mask_mix=1.1)
    with pytest.raises(ValueError):
        MaskWeightConfig(num_yolo_classes=0)
    with pytest.raises(ValueError):
        MaskWeightConfig(background_feature_consistency_loss_weight=-0.1)
    with pytest.raises(ValueError):
        MaskWeightConfig(background_token_consistency_loss_weight=-0.1)
    with pytest.raises(ValueError):
        MaskWeightConfig(target_background_contrastive_loss_weight=-0.1)
    with pytest.raises(ValueError):
        MaskWeightConfig(target_background_contrastive_margin=-0.1)


def test_target_token_attention_can_be_disabled_for_ablation():
    torch.manual_seed(0)
    config = MaskWeightConfig(
        adapter_hidden_dim=4,
        mask_dropout_p=0.0,
        mask_noise_p=0.0,
        use_target_tokens=True,
        use_target_token_attention=False,
    )
    adapter = MaskGuidedVisualAdapter(dim_model=8, config=config)

    features = torch.randn(2, 8, 5, 6)
    mask = torch.zeros(2, 1, 5, 6)
    mask[:, :, 1:4, 2:5] = 1.0

    _, target_tokens, target_pos_embed, mask_geometry_token, mask_geometry_pos_embed = adapter(features, mask)

    assert target_tokens.shape == (2, 2, 8)
    assert target_pos_embed.shape == (2, 1, 8)
    assert mask_geometry_token.shape == (1, 2, 8)
    assert mask_geometry_pos_embed.shape == (1, 1, 8)
    assert adapter.target_token_attention_scale is None
    assert adapter.target_object_perceiver is None
    assert adapter.latest_debug["target_token_attention_delta_ratio"] == pytest.approx(0.0)
    assert adapter.latest_debug["object_perceiver_delta_ratio"] == pytest.approx(0.0)
    assert adapter.latest_debug["object_perceiver_layers"] == pytest.approx(0.0)
    assert adapter.latest_debug["target_token_geometry_norm"] == pytest.approx(0.0)
    assert adapter.latest_debug["mask_geometry_token_norm"] >= 0.0


def test_mask_geometry_token_can_be_disabled_for_ablation():
    torch.manual_seed(0)
    config = MaskWeightConfig(
        adapter_hidden_dim=4,
        mask_dropout_p=0.0,
        mask_noise_p=0.0,
        use_target_tokens=True,
        use_mask_geometry_token=False,
    )
    adapter = MaskGuidedVisualAdapter(dim_model=8, config=config)

    features = torch.randn(2, 8, 5, 6)
    mask = torch.zeros(2, 1, 5, 6)
    mask[:, :, 1:4, 2:5] = 1.0

    _, target_tokens, target_pos_embed, mask_geometry_token, mask_geometry_pos_embed = adapter(features, mask)

    assert target_tokens.shape == (2, 2, 8)
    assert target_pos_embed.shape == (2, 1, 8)
    assert mask_geometry_token is None
    assert mask_geometry_pos_embed is None
    assert adapter.mask_geometry_token_proj is None
    assert adapter.mask_geometry_token_scale is None
    assert adapter.latest_debug["mask_geometry_token_norm"] == pytest.approx(0.0)


def test_object_weighted_mask_can_be_disabled_for_v42_behavior():
    torch.manual_seed(0)
    config = MaskWeightConfig(
        adapter_hidden_dim=4,
        mask_dropout_p=0.0,
        mask_noise_p=0.0,
        use_object_weighted_mask=False,
    )
    adapter = MaskGuidedVisualAdapter(dim_model=8, config=config)

    features = torch.randn(2, 8, 5, 6)
    union_mask = torch.zeros(2, 1, 5, 6)
    union_mask[:, :, 1:4, 2:5] = 1.0
    object_inputs = {
        "instance_masks": torch.zeros(2, 2, 1, 5, 6),
        "class_ids": torch.zeros(2, 2, dtype=torch.long),
        "confidences": torch.zeros(2, 2),
        "valid": torch.zeros(2, 2),
    }

    _, target_tokens, target_pos_embed, _, _ = adapter(
        features,
        union_mask,
        object_weight_inputs=object_inputs,
    )

    assert adapter.object_weight_net is None
    assert torch.allclose(adapter.latest_object_weighted_mask, union_mask)
    assert target_tokens.shape == (2, 2, 8)
    assert target_pos_embed.shape == (2, 1, 8)


def test_object_weighted_mask_preserves_v42_token_count_and_gets_gradients():
    torch.manual_seed(0)
    config = MaskWeightConfig(
        adapter_hidden_dim=4,
        mask_dropout_p=0.0,
        mask_noise_p=0.0,
        num_yolo_classes=4,
        max_object_weight_instances=3,
        object_weight_embed_dim=4,
        object_weight_hidden_dim=8,
        object_weight_attention_heads=2,
    )
    adapter = MaskGuidedVisualAdapter(dim_model=8, config=config)

    features = torch.randn(2, 8, 5, 6)
    union_mask = torch.zeros(2, 1, 5, 6)
    union_mask[:, :, 1:4, 2:5] = 1.0
    instance_masks = torch.zeros(2, 3, 1, 5, 6)
    instance_masks[:, 0, :, 1:4, 2:5] = 1.0
    instance_masks[:, 1, :, 0:2, 0:2] = 0.8
    object_inputs = {
        "instance_masks": instance_masks,
        "class_ids": torch.tensor([[1, 2, 0], [1, 2, 0]]),
        "confidences": torch.tensor([[0.9, 0.6, 0.0], [0.9, 0.6, 0.0]]),
        "valid": torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]),
    }

    _, target_tokens, target_pos_embed, mask_geometry_token, _ = adapter(
        features,
        union_mask,
        object_weight_inputs=object_inputs,
    )

    assert target_tokens.shape == (2, 2, 8)
    assert target_pos_embed.shape == (2, 1, 8)
    assert adapter.latest_consistency["target_tokens"].shape == (2, 2, 8)
    assert adapter.latest_debug["object_weight_valid_count"] == pytest.approx(2.0)
    assert adapter.latest_debug["object_weight_mean"] == pytest.approx(config.object_weight_init, abs=0.02)
    assert adapter.latest_debug["object_weight_final_mask_mean"] > 0.0
    assert adapter.latest_debug["object_weight_weighted_mask_mean"] > adapter.latest_debug["object_weight_floor"] * 0.0
    assert 1.0 <= adapter.latest_debug["object_weight_top_class"] <= 2.0
    assert adapter.latest_debug["object_weight_top_confidence"] > 0.0

    (target_tokens.mean() + mask_geometry_token.mean()).backward()
    assert adapter.object_weight_net.output[-1].weight.grad is not None
    assert adapter.object_weight_net.class_embed.weight.grad is not None
    assert torch.isfinite(adapter.object_weight_net.output[-1].weight.grad).all()
    assert torch.isfinite(adapter.object_weight_net.class_embed.weight.grad).all()


def test_object_weighted_mask_uses_base_first_blend_for_tokens():
    torch.manual_seed(0)
    config = MaskWeightConfig(
        adapter_hidden_dim=4,
        mask_dropout_p=0.0,
        mask_noise_p=0.0,
        num_yolo_classes=4,
        max_object_weight_instances=1,
        object_weight_embed_dim=4,
        object_weight_hidden_dim=8,
        object_weight_attention_heads=2,
        object_weight_union_floor=0.7,
        object_token_mask_mix=0.2,
        use_target_token_attention=False,
    )
    adapter = MaskGuidedVisualAdapter(dim_model=8, config=config)
    with torch.no_grad():
        adapter.object_weight_net.output[-1].weight.zero_()
        adapter.object_weight_net.output[-1].bias.fill_(torch.logit(torch.tensor(0.1)).item())

    features = torch.randn(2, 8, 5, 6)
    instance_masks = torch.zeros(2, 1, 1, 5, 6)
    instance_masks[:, 0, :, 1:4, 2:5] = 1.0
    union_mask = instance_masks.amax(dim=1)
    object_inputs = {
        "instance_masks": instance_masks,
        "class_ids": torch.ones(2, 1, dtype=torch.long),
        "confidences": torch.full((2, 1), 0.9),
        "valid": torch.ones(2, 1),
    }

    _, target_tokens, _, _, _ = adapter(features, union_mask, object_weight_inputs=object_inputs)

    union_mean = union_mask.mean().item()
    expected_object_mean = 0.1 * union_mean
    expected_base_mean = 0.7 * union_mean
    expected_object_token_mean = 0.8 * expected_base_mean + 0.2 * expected_object_mean
    assert target_tokens.shape == (2, 2, 8)
    assert adapter.latest_debug["object_weight_mean"] == pytest.approx(0.1, abs=1e-5)
    assert adapter.latest_debug["object_weight_weighted_mask_mean"] == pytest.approx(expected_object_mean, abs=1e-5)
    assert adapter.latest_debug["object_weight_object_token_mask_mean"] == pytest.approx(
        expected_object_token_mean,
        abs=1e-5,
    )
    assert adapter.latest_debug["object_weight_final_mask_mean"] == pytest.approx(expected_base_mean, abs=1e-5)
    assert adapter.latest_debug["object_weight_base_mask_mean"] == pytest.approx(expected_base_mean, abs=1e-5)
    assert adapter.latest_debug["object_weight_object_token_mix"] == pytest.approx(0.2)
    assert adapter.latest_debug["target_mask_mean"] == pytest.approx(expected_base_mean, abs=1e-5)
    assert adapter.latest_debug["object_token_target_mask_mean"] == pytest.approx(
        expected_object_token_mean,
        abs=1e-5,
    )
    assert adapter.latest_object_weighted_mask.mean().item() == pytest.approx(expected_base_mean, abs=1e-5)
    assert adapter.latest_target_token_debug["target_mask"].mean().item() == pytest.approx(
        expected_object_token_mean,
        abs=1e-5,
    )


def test_object_weighted_mask_dropout_is_shared_between_base_and_object_tokens():
    torch.manual_seed(0)
    config = MaskWeightConfig(
        adapter_hidden_dim=4,
        mask_dropout_p=1.0,
        mask_noise_p=0.0,
        num_yolo_classes=4,
        max_object_weight_instances=1,
        object_weight_embed_dim=4,
        object_weight_hidden_dim=8,
        object_weight_attention_heads=2,
        use_target_token_attention=False,
    )
    adapter = MaskGuidedVisualAdapter(dim_model=8, config=config)
    adapter.train()

    features = torch.randn(2, 8, 5, 6)
    instance_masks = torch.zeros(2, 1, 1, 5, 6)
    instance_masks[:, 0, :, 1:4, 2:5] = 1.0
    union_mask = instance_masks.amax(dim=1)
    object_inputs = {
        "instance_masks": instance_masks,
        "class_ids": torch.ones(2, 1, dtype=torch.long),
        "confidences": torch.full((2, 1), 0.9),
        "valid": torch.ones(2, 1),
    }

    _, target_tokens, _, mask_geometry_token, _ = adapter(features, union_mask, object_weight_inputs=object_inputs)

    assert target_tokens.shape == (2, 2, 8)
    assert torch.allclose(target_tokens, torch.zeros_like(target_tokens))
    assert torch.allclose(mask_geometry_token, torch.zeros_like(mask_geometry_token))
    assert adapter.latest_debug["target_mask_mean"] == pytest.approx(0.0)
    assert adapter.latest_debug["object_token_target_mask_mean"] == pytest.approx(0.0)
    assert adapter.latest_debug["reliability_mean"] == pytest.approx(0.0)
    assert adapter.latest_debug["object_token_reliability_mean"] == pytest.approx(0.0)


def test_object_weighted_mask_padding_slots_have_finite_gradients():
    torch.manual_seed(0)
    config = MaskWeightConfig(
        adapter_hidden_dim=4,
        mask_dropout_p=0.0,
        mask_noise_p=0.0,
        num_yolo_classes=4,
        max_object_weight_instances=8,
        object_weight_embed_dim=4,
        object_weight_hidden_dim=8,
        object_weight_attention_heads=2,
    )
    adapter = MaskGuidedVisualAdapter(dim_model=8, config=config)

    features = torch.randn(2, 8, 5, 6, requires_grad=True)
    instance_masks = torch.zeros(2, 8, 1, 5, 6)
    instance_masks[:, 0, :, 1:4, 2:5] = 0.9
    union_mask = instance_masks.amax(dim=1)
    object_inputs = {
        "instance_masks": instance_masks,
        "class_ids": torch.zeros(2, 8, dtype=torch.long),
        "confidences": torch.zeros(2, 8),
        "valid": torch.zeros(2, 8),
    }
    object_inputs["class_ids"][:, 0] = 1
    object_inputs["confidences"][:, 0] = 0.9
    object_inputs["valid"][:, 0] = 1.0

    guided_features, target_tokens, _, mask_geometry_token, _ = adapter(
        features,
        union_mask,
        object_weight_inputs=object_inputs,
    )
    loss = guided_features.square().mean() + target_tokens.square().mean() + mask_geometry_token.square().mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(features.grad).all()
    assert adapter.latest_debug["object_weight_valid_count"] == pytest.approx(1.0)
    assert adapter.latest_debug["object_weight_mean"] == pytest.approx(config.object_weight_init, abs=0.02)
    assert adapter.latest_debug["object_weight_final_mask_mean"] > 0.0
    for name, parameter in adapter.object_weight_net.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name


def test_empty_object_weight_inputs_do_not_inject_mask():
    torch.manual_seed(0)
    config = MaskWeightConfig(
        adapter_hidden_dim=4,
        mask_dropout_p=0.0,
        mask_noise_p=0.0,
        use_target_tokens=True,
        num_yolo_classes=4,
        max_object_weight_instances=2,
        object_weight_embed_dim=4,
        object_weight_hidden_dim=8,
        object_weight_attention_heads=2,
    )
    adapter = MaskGuidedVisualAdapter(dim_model=8, config=config)
    adapter.eval()

    features = torch.randn(2, 8, 5, 6)
    union_mask = torch.zeros(2, 1, 5, 6)
    object_inputs = {
        "instance_masks": torch.zeros(2, 2, 1, 5, 6),
        "class_ids": torch.zeros(2, 2, dtype=torch.long),
        "confidences": torch.zeros(2, 2),
        "valid": torch.zeros(2, 2),
    }

    guided_features, target_tokens, _, mask_geometry_token, _ = adapter(
        features,
        union_mask,
        object_weight_inputs=object_inputs,
    )

    assert torch.allclose(guided_features, features)
    assert torch.allclose(target_tokens, torch.zeros_like(target_tokens))
    assert torch.allclose(mask_geometry_token, torch.zeros_like(mask_geometry_token))
    assert torch.allclose(adapter.latest_object_weighted_mask, union_mask)
    assert adapter.latest_debug["object_weight_valid_count"] == pytest.approx(0.0)
    assert adapter.latest_debug["object_weight_mean"] == pytest.approx(0.0)


def test_yolo_results_to_object_weight_inputs_uses_box_fallback_and_confidence_order():
    class FakeBoxes:
        def __init__(self):
            self.conf = torch.tensor([0.2, 0.9, 0.5])
            self.cls = torch.tensor([3, 1, 2])
            self.xyxy = torch.tensor(
                [
                    [0.0, 0.0, 2.0, 2.0],
                    [1.0, 1.0, 4.0, 4.0],
                    [2.0, 0.0, 5.0, 2.0],
                ]
            )

        def __len__(self):
            return int(self.conf.numel())

    result = SimpleNamespace(orig_shape=(6, 6), masks=None, boxes=FakeBoxes())

    object_inputs = yolo_results_to_object_weight_inputs([result], max_instances=2, kernel_size=1)

    assert object_inputs["instance_masks"].shape == (1, 2, 1, 6, 6)
    assert object_inputs["confidences"].flatten().tolist() == pytest.approx([0.9, 0.5])
    assert object_inputs["class_ids"].flatten().tolist() == [1, 2]
    assert object_inputs["valid"].flatten().tolist() == pytest.approx([1.0, 1.0])
    assert object_inputs["instance_masks"][0, 0].max().item() == pytest.approx(0.9)
    assert object_inputs["instance_masks"][0, 1].max().item() == pytest.approx(0.5)


def test_region_attention_reliability_gate_falls_back_on_empty_mask():
    torch.manual_seed(0)
    config = MaskWeightConfig(
        adapter_hidden_dim=4,
        mask_dropout_p=0.0,
        mask_noise_p=0.0,
        use_target_tokens=False,
    )
    adapter = MaskGuidedVisualAdapter(dim_model=8, config=config)
    adapter.eval()

    features = torch.randn(2, 8, 5, 6)
    mask = torch.zeros(2, 1, 5, 6)

    guided_features, target_tokens, target_pos_embed, mask_geometry_token, mask_geometry_pos_embed = adapter(
        features,
        mask,
    )

    assert torch.allclose(guided_features, features)
    assert target_tokens is None
    assert target_pos_embed is None
    assert torch.allclose(mask_geometry_token, torch.zeros_like(mask_geometry_token))
    assert mask_geometry_pos_embed.shape == (1, 1, 8)
    assert adapter.latest_debug["reliability_mean"] == pytest.approx(0.0)
    assert adapter.latest_debug["region_attention_delta_ratio"] == pytest.approx(0.0)
    assert adapter.latest_debug["mask_geometry_token_norm"] == pytest.approx(0.0)


def test_target_token_attention_does_not_inject_empty_mask():
    torch.manual_seed(0)
    config = MaskWeightConfig(
        adapter_hidden_dim=4,
        mask_dropout_p=0.0,
        mask_noise_p=0.0,
        use_target_tokens=True,
    )
    adapter = MaskGuidedVisualAdapter(dim_model=8, config=config)
    adapter.eval()

    features = torch.randn(2, 8, 5, 6)
    mask = torch.zeros(2, 1, 5, 6)

    guided_features, target_tokens, target_pos_embed, mask_geometry_token, mask_geometry_pos_embed = adapter(
        features,
        mask,
    )

    assert torch.allclose(guided_features, features)
    assert torch.allclose(target_tokens, torch.zeros_like(target_tokens))
    assert target_pos_embed.shape == (2, 1, 8)
    assert torch.allclose(mask_geometry_token, torch.zeros_like(mask_geometry_token))
    assert mask_geometry_pos_embed.shape == (1, 1, 8)
    assert adapter.latest_debug["reliability_mean"] == pytest.approx(0.0)
    assert adapter.latest_debug["target_token_attention_delta_ratio"] == pytest.approx(0.0)
    assert adapter.latest_debug["object_perceiver_delta_ratio"] == pytest.approx(0.0)
    assert adapter.latest_debug["target_token_geometry_norm"] == pytest.approx(0.0)
    assert adapter.latest_debug["mask_geometry_token_norm"] == pytest.approx(0.0)


def test_target_background_contrastive_loss_has_debug_metrics():
    policy = ACTPolicy.__new__(ACTPolicy)
    policy.training = True
    policy.config = SimpleNamespace(use_mask_weight=True, mw_config=MaskWeightConfig())
    reference = {
        "cam": {
            "target_tokens": torch.ones(2, 2, 4),
            "target_feature": torch.ones(2, 4),
            "background_feature": -torch.ones(2, 4),
        }
    }

    loss, debug = policy._compute_mask_weight_target_token_contrastive(reference, record_debug=True)

    assert loss is not None
    assert debug["mask_weight/target_token/contrastive_pairs"] == pytest.approx(1.0)
    assert debug["mask_weight/target_token/contrastive_loss"] == pytest.approx(0.0)
    assert debug["mask_weight/target_token/target_cosine"] > debug["mask_weight/target_token/background_cosine"]


def test_representation_consistency_uses_normalized_loss_with_raw_debug():
    policy = ACTPolicy.__new__(ACTPolicy)
    policy.config = SimpleNamespace(use_mask_weight=True, mw_config=MaskWeightConfig())
    reference = {
        "cam": {
            "target_feature": torch.ones(2, 4),
            "target_tokens": torch.ones(2, 2, 4),
        }
    }
    augmented = {
        "cam": {
            "target_feature": torch.ones(2, 4) * 1000.0,
            "target_tokens": torch.ones(2, 2, 4) * 1000.0,
        }
    }

    loss, debug = policy._compute_mask_weight_representation_consistency(
        reference,
        augmented,
        record_debug=True,
    )

    assert loss is not None
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0)
    assert debug["mask_weight/consistency/feature_loss"] == pytest.approx(0.0)
    assert debug["mask_weight/consistency/token_loss"] == pytest.approx(0.0)
    assert debug["mask_weight/consistency/feature_raw_l1"] > 900.0
    assert debug["mask_weight/consistency/token_raw_l1"] > 900.0


def test_mask_guided_background_augmentation_preserves_target_and_changes_background():
    torch.manual_seed(0)
    config = MaskWeightConfig(
        background_aug_p=1.0,
        background_aug_context_dilation=0,
        background_aug_keep_threshold=0.5,
        background_aug_mode="random_color",
    )
    images = torch.zeros(2, 3, 4, 4)
    images[:, :, 1:3, 1:3] = 0.7
    masks = torch.zeros(2, 1, 4, 4)
    masks[:, :, 1:3, 1:3] = 1.0

    augmented, debug = make_mask_guided_background_augmentation(images, masks, config)

    assert torch.allclose(augmented[:, :, 1:3, 1:3], images[:, :, 1:3, 1:3])
    assert not torch.allclose(augmented * (1.0 - masks), images * (1.0 - masks))
    assert debug["applied_ratio"].item() == pytest.approx(1.0)
    assert debug["background_replaced_ratio"].item() == pytest.approx(0.75)
    assert debug["background_effective_change_ratio"].item() == pytest.approx(0.75)
    assert debug["strength_mean"].item() == pytest.approx(1.0)
    assert debug["strength_min"].item() == pytest.approx(1.0)
    assert debug["strength_max"].item() == pytest.approx(1.0)


def test_mask_guided_background_augmentation_skips_samples_without_mask():
    torch.manual_seed(0)
    config = MaskWeightConfig(background_aug_p=1.0)
    images = torch.rand(2, 3, 4, 4)
    masks = torch.zeros(2, 1, 4, 4)

    augmented, debug = make_mask_guided_background_augmentation(images, masks, config)

    assert torch.allclose(augmented, images)
    assert debug["applied_ratio"].item() == pytest.approx(0.0)


def test_mask_guided_background_augmentation_strength_override():
    torch.manual_seed(0)
    config = MaskWeightConfig(
        background_aug_p=1.0,
        background_aug_context_dilation=0,
        background_aug_keep_threshold=0.5,
        background_aug_mode="random_color",
    )
    images = torch.zeros(2, 3, 4, 4)
    masks = torch.zeros(2, 1, 4, 4)
    masks[:, :, 1:3, 1:3] = 1.0

    _, debug = make_mask_guided_background_augmentation(
        images,
        masks,
        config,
        min_strength_override=0.8,
        max_strength_override=0.8,
    )

    assert debug["strength_min"].item() == pytest.approx(0.8)
    assert debug["strength_max"].item() == pytest.approx(0.8)
    assert debug["strength_mean"].item() == pytest.approx(0.8)
    assert debug["background_effective_change_ratio"].item() == pytest.approx(0.6)
