import pytest
import torch

from lerobot.policies.customACT.mask_weight.configuration_mask_weight import MaskWeightConfig
from lerobot.policies.customACT.mask_weight.mask_weight import (
    MaskGuidedVisualAdapter,
    make_mask_guided_background_augmentation,
)


def test_mask_weight_config_v4_direct_defaults():
    config = MaskWeightConfig()

    assert config.use_target_tokens is True
    assert config.num_target_tokens == 2
    assert config.gate_init == pytest.approx(0.2)
    assert config.region_attention_gate_init == pytest.approx(0.2)
    assert config.region_attention_context_weight == pytest.approx(0.3)
    assert config.region_attention_background_weight == pytest.approx(0.0)
    assert config.reliability_floor == pytest.approx(0.6)
    assert config.mask_noise_p == pytest.approx(0.02)
    assert config.mask_noise_dropout_p == pytest.approx(0.0)
    assert config.background_feature_consistency_loss_weight == pytest.approx(0.03)
    assert config.background_token_consistency_loss_weight == pytest.approx(0.03)


def test_mask_guided_visual_adapter_shapes_and_gradient():
    torch.manual_seed(0)
    config = MaskWeightConfig(adapter_hidden_dim=4, mask_dropout_p=0.0, mask_noise_p=0.0, use_target_tokens=False)
    adapter = MaskGuidedVisualAdapter(dim_model=8, config=config)

    features = torch.randn(2, 8, 5, 6, requires_grad=True)
    mask = torch.zeros(2, 1, 5, 6)
    mask[:, :, 1:4, 2:5] = 0.8

    guided_features, target_tokens, target_pos_embed = adapter(features, mask)

    assert guided_features.shape == features.shape
    assert target_tokens is None
    assert target_pos_embed is None
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

    guided_features, target_tokens, target_pos_embed = adapter(features, mask)

    assert guided_features.shape == features.shape
    assert target_tokens.shape == (3, 2, 8)
    assert target_pos_embed.shape == (3, 1, 8)
    assert adapter.latest_consistency["target_feature"].shape == (2, 8)
    assert adapter.latest_consistency["target_tokens"].shape == (3, 2, 8)


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
        MaskWeightConfig(background_feature_consistency_loss_weight=-0.1)
    with pytest.raises(ValueError):
        MaskWeightConfig(background_token_consistency_loss_weight=-0.1)


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

    guided_features, target_tokens, target_pos_embed = adapter(features, mask)

    assert torch.allclose(guided_features, features)
    assert target_tokens is None
    assert target_pos_embed is None
    assert adapter.latest_debug["reliability_mean"] == pytest.approx(0.0)
    assert adapter.latest_debug["region_attention_delta_ratio"] == pytest.approx(0.0)


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
