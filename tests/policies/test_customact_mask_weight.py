import pytest
import torch

from lerobot.policies.customACT.mask_weight.configuration_mask_weight import MaskWeightConfig
from lerobot.policies.customACT.mask_weight.mask_weight import (
    MaskGuidedVisualAdapter,
    make_mask_guided_background_augmentation,
)


def test_mask_guided_visual_adapter_shapes_and_gradient():
    torch.manual_seed(0)
    config = MaskWeightConfig(adapter_hidden_dim=4, mask_dropout_p=0.0, use_target_tokens=False)
    adapter = MaskGuidedVisualAdapter(dim_model=8, config=config)

    features = torch.randn(2, 8, 5, 6, requires_grad=True)
    mask = torch.zeros(2, 1, 5, 6)
    mask[:, :, 1:4, 2:5] = 0.8

    guided_features, target_tokens, target_pos_embed = adapter(features, mask)

    assert guided_features.shape == features.shape
    assert target_tokens is None
    assert target_pos_embed is None

    guided_features.mean().backward()
    assert adapter.gate_scale.grad is not None
    assert adapter.background_suppress_scale.grad is not None


def test_mask_guided_visual_adapter_target_tokens():
    torch.manual_seed(0)
    config = MaskWeightConfig(
        adapter_hidden_dim=4,
        mask_dropout_p=0.0,
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


def test_mask_guided_visual_adapter_region_modulation():
    config = MaskWeightConfig(
        mask_dropout_p=0.0,
        use_spatial_embedding=False,
        use_residual_gate=False,
        use_target_tokens=False,
        context_dilation=0,
        target_boost_init=0.2,
        context_boost_init=0.0,
        background_suppress_init=0.2,
    )
    adapter = MaskGuidedVisualAdapter(dim_model=1, config=config)

    features = torch.ones(1, 1, 3, 3)
    mask = torch.zeros(1, 1, 3, 3)
    mask[:, :, 1, 1] = 1.0

    guided_features, _, _ = adapter(features, mask)

    assert guided_features[0, 0, 1, 1].item() == pytest.approx(1.2)
    assert guided_features[0, 0, 0, 0].item() == pytest.approx(0.8)
    assert adapter.latest_debug["region_delta_ratio"] > 0.0


def test_mask_weight_config_rejects_invalid_mode():
    with pytest.raises(ValueError):
        MaskWeightConfig(mode="bad_mode")


def test_mask_guided_background_augmentation_preserves_target_and_changes_background():
    torch.manual_seed(0)
    config = MaskWeightConfig(
        background_aug_p=1.0,
        background_aug_context_dilation=0,
        background_aug_keep_threshold=0.5,
        background_aug_mode="random_color",
        background_aug_min_strength=1.0,
        background_aug_max_strength=1.0,
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
    assert debug["strength_mean"].item() == pytest.approx(1.0)


def test_mask_guided_background_augmentation_skips_samples_without_mask():
    torch.manual_seed(0)
    config = MaskWeightConfig(background_aug_p=1.0)
    images = torch.rand(2, 3, 4, 4)
    masks = torch.zeros(2, 1, 4, 4)

    augmented, debug = make_mask_guided_background_augmentation(images, masks, config)

    assert torch.allclose(augmented, images)
    assert debug["applied_ratio"].item() == pytest.approx(0.0)
