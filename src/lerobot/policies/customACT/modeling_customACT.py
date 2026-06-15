#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Action Chunking Transformer Policy

As per Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (https://huggingface.co/papers/2304.13705).
The majority of changes here involve removing unused code, unifying naming, and adding helpful comments.
"""

import math
from collections import deque
from collections.abc import Callable
from itertools import chain

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d
from lerobot.policies.dino_act.dino_backbone import DinoV2Backbone

from lerobot.policies.customACT.configuration_customACT import ACTConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE, HIS_OBS_STATES
#新增：自己的import
from lerobot.policies.customACT.history_obs_state.modeling_history_obs import HistoryObsStateEmbedding
from lerobot.policies.dino_act.backbone_res import ResNet18Backbone, get_custom_backbone
from lerobot.policies.dino_act.convnext import ConvNeXtBackbone
from lerobot.policies.dino_act.convnext_frame import ConvNeXtBackbone1
from lerobot.policies.customACT.segment_understanding.modeling_segment_understanding import SegmentUnderstandingEmbedding
from lerobot.policies.customACT.segment_understanding.utils.kinematics import SimpleKinematics
from lerobot.policies.customACT.segment_understanding.utils.yolo_data_processer import YoloDataProcessor
from lerobot.policies.customACT.mask_weight.mask_weight import (
    MaskGuidedVisualAdapter,
    make_mask_guided_background_augmentation,
    yolo_result_to_soft_mask,
    yolo_results_to_instance_masks,
)
# from lerobot.policies.customACT.segment_understanding.utils.denormalize import denormalize_img_with_mean_stats,denormalize_obs_and_angle_to_rad
# 由于想要未经初始化的数据，需要用到这个
from lerobot.processor import PolicyProcessorPipeline
from typing import Any
from lerobot.processor.normalize_processor import NormalizerProcessorStep
from lerobot.configs.types import FeatureType

_ACT_FORCED_LATENT_SAMPLE = "_act_forced_latent_sample"
_MASK_WEIGHT_FORCED_MASKS = "_mask_weight_forced_masks"
_MASK_WEIGHT_FORCED_INSTANCE_MASKS = "_mask_weight_forced_instance_masks"
_MASK_WEIGHT_SKIP_DEBUG = "_mask_weight_skip_debug"

class ACTPolicy(PreTrainedPolicy):
    """
    Action Chunking Transformer Policy as per Learning Fine-Grained Bimanual Manipulation with Low-Cost
    Hardware (paper: https://huggingface.co/papers/2304.13705, code: https://github.com/tonyzhaozh/act)
    """

    config_class = ACTConfig
    name = "act"

    def __init__(
        self,
        config: ACTConfig,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = ACT(config)
        self.model.enable_debug_visualization = False

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def get_optim_params(self) -> dict:
        # TODO(aliberts, rcadene): As of now, lr_backbone == lr
        # Should we remove this and just `return self.parameters()`?
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        """This should be called whenever the environment is reset."""
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()  # keeping the policy in eval mode as it could be set to train mode while queue is consumed

        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            action = self.temporal_ensembler.update(actions)
            return action

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor: # 实际运行时的100步动作预测
        """Predict a chunk of actions given environment observations."""
        self.eval()

        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions = self.model(batch)[0]
        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]: # ACTPolicy的前向传播，里面调用了self.model
        """Run the batch through the model and compute the loss for training or validation."""
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        #DEBUG 输出batch的结构
        # for key,tensor in batch.items(): # print出batch的内容，方便调试
        #     print(f"{key}: {tensor.shape if isinstance(tensor, Tensor) else type(tensor)}")

        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        l1_loss = (
            F.l1_loss(batch[ACTION], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.item()}
        if self.config.use_vae:
            # Calculate Dₖₗ(latent_pdf || standard_normal). Note: After computing the KL-divergence for
            # each dimension independently, we sum over the latent dimension to get the total
            # KL-divergence per batch element, then take the mean over the batch.
            # (See App. B of https://huggingface.co/papers/1312.6114 for more details).
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss

        record_mask_weight_debug = self._should_record_mask_weight_debug()
        mask_weight_debug = (
            dict(getattr(self.model, "latest_mask_weight_debug", None) or {})
            if record_mask_weight_debug
            else {}
        )
        mask_weight_masks = dict(getattr(self.model, "latest_mask_weight_masks", None) or {})
        mask_weight_instance_masks = dict(getattr(self.model, "latest_mask_weight_instance_masks", None) or {})
        mask_weight_consistency_refs = dict(getattr(self.model, "latest_mask_weight_consistency", None) or {})
        background_debug: dict[str, float | bool] = {}
        target_token_loss, target_token_debug = self._compute_mask_weight_target_token_contrastive(
            mask_weight_consistency_refs,
            record_debug=record_mask_weight_debug,
        )
        if target_token_loss is not None:
            loss = loss + target_token_loss

        if self._should_use_mask_weight_background_consistency():
            augmented = self._build_mask_weight_background_augmented_batch(
                batch,
                mask_weight_masks,
                mask_weight_instance_masks,
            )
            if augmented is not None:
                aug_batch, background_debug = augmented
                latent_sample = getattr(self.model, "latest_latent_sample", None)
                if latent_sample is not None:
                    aug_batch[_ACT_FORCED_LATENT_SAMPLE] = latent_sample.detach()

                actions_aug, _ = self.model(aug_batch)
                mask_weight_consistency_aug = dict(
                    getattr(self.model, "latest_mask_weight_consistency", None) or {}
                )

                # Keep debug/Rerun data from the real image path; the augmented pass
                # is only a training constraint.
                self.model.latest_mask_weight_debug = dict(mask_weight_debug)
                self.model.latest_mask_weight_masks = dict(mask_weight_masks)
                self.model.latest_mask_weight_instance_masks = dict(mask_weight_instance_masks)
                self.model.latest_mask_weight_consistency = self._detach_mask_weight_consistency(
                    mask_weight_consistency_refs
                )

                action_delta = F.l1_loss(actions_aug, actions_hat.detach(), reduction="none")
                valid_action = ~batch["action_is_pad"].unsqueeze(-1)
                consistency_loss = (action_delta * valid_action).mean()
                consistency_weight = float(self.config.mw_config.background_consistency_loss_weight)
                loss = loss + consistency_loss * consistency_weight
                representation_loss, representation_debug = self._compute_mask_weight_representation_consistency(
                    mask_weight_consistency_refs,
                    mask_weight_consistency_aug,
                    record_debug=record_mask_weight_debug,
                )
                if representation_loss is not None:
                    loss = loss + representation_loss

                if record_mask_weight_debug:
                    masked_delta = action_delta.detach() * valid_action
                    valid_denom = (valid_action.sum() * action_delta.shape[-1]).clamp(min=1)
                    action_delta_l1 = masked_delta.sum() / valid_denom
                    action_delta_max = masked_delta.max()
                    consistency_loss_value = float(consistency_loss.detach().item())
                    consistency_weighted_value = float((consistency_loss.detach() * consistency_weight).item())
                    action_delta_l1_value = float(action_delta_l1.detach().item())
                    action_delta_max_value = float(action_delta_max.detach().item())
                    background_debug["mask_weight/bg_aug/consistency_loss"] = consistency_loss_value
                    background_debug["mask_weight/bg_aug/consistency_weight"] = consistency_weight
                    background_debug["mask_weight/bg_aug/action_delta_l1"] = action_delta_l1_value
                    background_debug["mask_weight/bg_aug/action_delta_max"] = action_delta_max_value
                    background_debug["mask_weight/consistency/enabled"] = 1.0
                    background_debug["mask_weight/consistency/weight"] = consistency_weight
                    background_debug["mask_weight/consistency/loss"] = consistency_loss_value
                    background_debug["mask_weight/consistency/weighted_loss"] = consistency_weighted_value
                    background_debug["mask_weight/consistency/action_delta_l1"] = action_delta_l1_value
                    background_debug["mask_weight/consistency/action_delta_max"] = action_delta_max_value
                    background_debug["mask_weight/consistency/latent_reused"] = float(latent_sample is not None)
                    background_debug.update(representation_debug)
            elif record_mask_weight_debug:
                background_debug = {
                    "mask_weight/bg_aug/enabled": True,
                    "mask_weight/bg_aug/applied_ratio": 0.0,
                    "mask_weight/bg_aug/skipped": True,
                }

        if mask_weight_debug:
            loss_dict.update(mask_weight_debug)
        if background_debug:
            loss_dict.update(background_debug)
        if target_token_debug:
            loss_dict.update(target_token_debug)

        if mask_weight_consistency_refs:
            self.model.latest_mask_weight_consistency = self._detach_mask_weight_consistency(
                mask_weight_consistency_refs
            )

        return loss, loss_dict

    def _should_record_mask_weight_debug(self) -> bool:
        return self.config.use_mask_weight and bool(getattr(self.config.mw_config, "record_debug_log", True))

    def _mask_weight_background_consistency_weight_sum(self) -> float:
        mw_config = self.config.mw_config
        return (
            max(0.0, float(getattr(mw_config, "background_consistency_loss_weight", 0.0)))
            + max(0.0, float(getattr(mw_config, "background_feature_consistency_loss_weight", 0.0)))
            + max(0.0, float(getattr(mw_config, "background_token_consistency_loss_weight", 0.0)))
        )

    @staticmethod
    def _detach_mask_weight_consistency(
        consistency: dict[str, dict[str, Tensor]],
    ) -> dict[str, dict[str, Tensor]]:
        detached: dict[str, dict[str, Tensor]] = {}
        for img_key, data in consistency.items():
            if not isinstance(data, dict):
                continue
            detached[img_key] = {
                name: value.detach()
                for name, value in data.items()
                if isinstance(value, Tensor)
            }
        return detached

    def _compute_mask_weight_target_token_contrastive(
        self,
        consistency: dict[str, dict[str, Tensor]],
        record_debug: bool,
    ) -> tuple[Tensor | None, dict[str, float]]:
        mw_config = self.config.mw_config
        weight = float(getattr(mw_config, "target_background_contrastive_loss_weight", 0.0))
        margin = float(getattr(mw_config, "target_background_contrastive_margin", 0.0))
        if (
            not self.training
            or not self.config.use_mask_weight
            or weight <= 0.0
            or not consistency
        ):
            return None, {}

        losses = []
        target_cosines = []
        background_cosines = []
        for data in consistency.values():
            if not isinstance(data, dict):
                continue
            target_tokens = data.get("target_tokens")
            target_feature = data.get("target_feature")
            background_feature = data.get("background_feature")
            if (
                not isinstance(target_tokens, Tensor)
                or not isinstance(target_feature, Tensor)
                or not isinstance(background_feature, Tensor)
                or target_tokens.ndim != 3
                or target_tokens.shape[0] < 1
            ):
                continue

            target_token = target_tokens[0]
            if target_token.shape != target_feature.shape or target_token.shape != background_feature.shape:
                continue

            target_cosine = F.cosine_similarity(target_token, target_feature.detach(), dim=-1)
            background_cosine = F.cosine_similarity(target_token, background_feature.detach(), dim=-1)
            losses.append(F.relu(margin + background_cosine - target_cosine).mean())
            target_cosines.append(target_cosine.detach().mean())
            background_cosines.append(background_cosine.detach().mean())

        if not losses:
            if record_debug:
                return None, {
                    "mask_weight/target_token/contrastive_weight": weight,
                    "mask_weight/target_token/contrastive_pairs": 0.0,
                }
            return None, {}

        contrastive_loss = torch.stack(losses).mean()
        weighted_loss = contrastive_loss * weight
        if not record_debug:
            return weighted_loss, {}

        debug = {
            "mask_weight/target_token/contrastive_loss": float(contrastive_loss.detach().item()),
            "mask_weight/target_token/contrastive_weight": weight,
            "mask_weight/target_token/contrastive_weighted_loss": float(weighted_loss.detach().item()),
            "mask_weight/target_token/contrastive_margin": margin,
            "mask_weight/target_token/contrastive_pairs": float(len(losses)),
            "mask_weight/target_token/target_cosine": float(torch.stack(target_cosines).mean().item()),
            "mask_weight/target_token/background_cosine": float(torch.stack(background_cosines).mean().item()),
        }
        return weighted_loss, debug

    def _should_use_mask_weight_background_consistency(self) -> bool:
        mw_config = self.config.mw_config
        return (
            self.training
            and self.config.use_mask_weight
            and bool(self.config.image_features)
            and mw_config.use_background_augmentation
            and mw_config.use_background_consistency
            and mw_config.background_aug_p > 0.0
            and self._mask_weight_background_consistency_weight_sum() > 0.0
            and getattr(self.model, "preprocessor", None) is not None
        )

    def _compute_mask_weight_representation_consistency(
        self,
        reference: dict[str, dict[str, Tensor]],
        augmented: dict[str, dict[str, Tensor]],
        record_debug: bool,
    ) -> tuple[Tensor | None, dict[str, float]]:
        mw_config = self.config.mw_config
        feature_weight = float(getattr(mw_config, "background_feature_consistency_loss_weight", 0.0))
        token_weight = float(getattr(mw_config, "background_token_consistency_loss_weight", 0.0))
        total_loss: Tensor | None = None
        debug: dict[str, float] = {}

        def _add_loss(current: Tensor | None, term: Tensor) -> Tensor:
            return term if current is None else current + term

        if feature_weight > 0.0:
            feature_losses = []
            for img_key, ref_data in reference.items():
                aug_data = augmented.get(img_key, {})
                ref_feature = ref_data.get("target_feature") if isinstance(ref_data, dict) else None
                aug_feature = aug_data.get("target_feature") if isinstance(aug_data, dict) else None
                if (
                    isinstance(ref_feature, Tensor)
                    and isinstance(aug_feature, Tensor)
                    and ref_feature.shape == aug_feature.shape
                ):
                    feature_losses.append(F.l1_loss(aug_feature, ref_feature.detach()))

            if feature_losses:
                feature_loss = torch.stack(feature_losses).mean()
                total_loss = _add_loss(total_loss, feature_loss * feature_weight)
                if record_debug:
                    debug["mask_weight/consistency/feature_loss"] = float(feature_loss.detach().item())
                    debug["mask_weight/consistency/feature_weight"] = feature_weight
                    debug["mask_weight/consistency/feature_weighted_loss"] = float(
                        (feature_loss.detach() * feature_weight).item()
                    )
                    debug["mask_weight/consistency/feature_pairs"] = float(len(feature_losses))
            elif record_debug:
                debug["mask_weight/consistency/feature_weight"] = feature_weight
                debug["mask_weight/consistency/feature_pairs"] = 0.0

        if token_weight > 0.0:
            token_losses = []
            for img_key, ref_data in reference.items():
                aug_data = augmented.get(img_key, {})
                ref_tokens = ref_data.get("target_tokens") if isinstance(ref_data, dict) else None
                aug_tokens = aug_data.get("target_tokens") if isinstance(aug_data, dict) else None
                if (
                    isinstance(ref_tokens, Tensor)
                    and isinstance(aug_tokens, Tensor)
                    and ref_tokens.shape == aug_tokens.shape
                ):
                    token_losses.append(F.l1_loss(aug_tokens, ref_tokens.detach()))

            if token_losses:
                token_loss = torch.stack(token_losses).mean()
                total_loss = _add_loss(total_loss, token_loss * token_weight)
                if record_debug:
                    debug["mask_weight/consistency/token_loss"] = float(token_loss.detach().item())
                    debug["mask_weight/consistency/token_weight"] = token_weight
                    debug["mask_weight/consistency/token_weighted_loss"] = float(
                        (token_loss.detach() * token_weight).item()
                    )
                    debug["mask_weight/consistency/token_pairs"] = float(len(token_losses))
            elif record_debug:
                debug["mask_weight/consistency/token_weight"] = token_weight
                debug["mask_weight/consistency/token_pairs"] = 0.0

        return total_loss, debug

    def _build_mask_weight_background_augmented_batch(
        self,
        batch: dict[str, Tensor],
        masks: dict[str, Tensor],
        instance_masks: dict[str, Tensor] | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, float | bool]] | None:
        if not masks:
            return None

        preprocessor = getattr(self.model, "preprocessor", None)
        if preprocessor is None or len(getattr(preprocessor, "steps", [])) <= 3:
            return None
        norm_step: NormalizerProcessorStep = preprocessor.steps[3]

        aug_batch = dict(batch)
        forced_masks: dict[str, Tensor] = {}
        forced_instance_masks: dict[str, Tensor] = {}
        debug: dict[str, float | bool] = {"mask_weight/bg_aug/enabled": True}
        metric_totals: dict[str, float] = {}
        num_cameras = 0
        sample_apply_mask: Tensor | None = None

        for img_key in self.config.image_features:
            if img_key not in batch or img_key not in masks:
                continue

            img = batch[img_key]
            if sample_apply_mask is None:
                sample_apply_mask = (
                    torch.rand(img.shape[0], 1, 1, 1, device=img.device) < self.config.mw_config.background_aug_p
                )

            img_01 = norm_step._apply_transform(img, img_key, FeatureType.VISUAL, inverse=True)
            img_01 = torch.clamp(img_01, 0.0, 1.0)
            mask = masks[img_key].to(device=img.device)

            aug_img_01, camera_debug = make_mask_guided_background_augmentation(
                img_01,
                mask,
                self.config.mw_config,
                apply_mask=sample_apply_mask,
                min_strength_override=self.config.mw_config.background_consistency_aug_min_strength,
                max_strength_override=self.config.mw_config.background_consistency_aug_max_strength,
            )
            aug_batch[img_key] = norm_step._apply_transform(
                aug_img_01,
                img_key,
                FeatureType.VISUAL,
                inverse=False,
            )
            forced_masks[img_key] = mask.detach()
            if isinstance(instance_masks, dict) and img_key in instance_masks:
                forced_instance_masks[img_key] = instance_masks[img_key].detach()

            cam_name = img_key.replace(f"{OBS_IMAGES}.", "")
            for name, tensor_value in camera_debug.items():
                value = float(tensor_value.detach().item())
                debug[f"mask_weight/bg_aug/{cam_name}/{name}"] = value
                metric_totals[name] = metric_totals.get(name, 0.0) + value
            num_cameras += 1

        if not forced_masks:
            return None

        if self.config.image_features:
            aug_batch[OBS_IMAGES] = [aug_batch[key] for key in self.config.image_features]
        aug_batch[_MASK_WEIGHT_FORCED_MASKS] = forced_masks
        aug_batch[_MASK_WEIGHT_FORCED_INSTANCE_MASKS] = forced_instance_masks
        aug_batch[_MASK_WEIGHT_SKIP_DEBUG] = True

        debug["mask_weight/bg_aug/cameras"] = float(num_cameras)
        for name, value in metric_totals.items():
            debug[f"mask_weight/bg_aug/{name}"] = value / max(1, num_cameras)

        return aug_batch, debug

    # 专门给yolo和fk用的预处理器设置函数，它们需要没有经过归一化的数据，然而lerobot传入的batch已经经过归一化了
    def set_preprocessor(self, preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,):
        self.model.preprocessor = preprocessor

    def enable_debug_visualization(self, enable: bool = True) -> None:
        self.model.enable_debug_visualization = enable
        if not enable:
            self.model.latest_yolo_debug_overlays = {}
            self.model.pending_yolo_debug_overlays = {}

    def get_debug_observation_images(self) -> dict[str, np.ndarray]:
        overlays = getattr(self.model, "pending_yolo_debug_overlays", None)
        mask_weight_debug = getattr(self.model, "latest_mask_weight_debug", None)
        if not overlays and not mask_weight_debug:
            return {}
        output = dict(overlays) if overlays else {}
        if overlays is not None:
            overlays.clear()
        if mask_weight_debug:
            for key, value in mask_weight_debug.items():
                if isinstance(value, (int, float, bool)):
                    output[f"custom.{key.replace('/', '.')}"] = np.array(value, dtype=np.float32)
        return output
        

class ACTTemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        """Temporal ensembling as described in Algorithm 2 of https://huggingface.co/papers/2304.13705.

        The weights are calculated as wᵢ = exp(-temporal_ensemble_coeff * i) where w₀ is the oldest action.
        They are then normalized to sum to 1 by dividing by Σwᵢ. Here's some intuition around how the
        coefficient works:
            - Setting it to 0 uniformly weighs all actions.
            - Setting it positive gives more weight to older actions.
            - Setting it negative gives more weight to newer actions.
        NOTE: The default value for `temporal_ensemble_coeff` used by the original ACT work is 0.01. This
        results in older actions being weighed more highly than newer actions (the experiments documented in
        https://github.com/huggingface/lerobot/pull/319 hint at why highly weighing new actions might be
        detrimental: doing so aggressively may diminish the benefits of action chunking).

        Here we use an online method for computing the average rather than caching a history of actions in
        order to compute the average offline. For a simple 1D sequence it looks something like:

        ```
        import torch

        seq = torch.linspace(8, 8.5, 100)
        print(seq)

        m = 0.01
        exp_weights = torch.exp(-m * torch.arange(len(seq)))
        print(exp_weights)

        # Calculate offline
        avg = (exp_weights * seq).sum() / exp_weights.sum()
        print("offline", avg)

        # Calculate online
        for i, item in enumerate(seq):
            if i == 0:
                avg = item
                continue
            avg *= exp_weights[:i].sum()
            avg += item * exp_weights[i]
            avg /= exp_weights[: i + 1].sum()
        print("online", avg)
        ```
        """
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        """Resets the online computation variables."""
        self.ensembled_actions = None
        # (chunk_size,) count of how many actions are in the ensemble for each time step in the sequence.
        self.ensembled_actions_count = None

    def update(self, actions: Tensor) -> Tensor:
        """
        Takes a (batch, chunk_size, action_dim) sequence of actions, update the temporal ensemble for all
        time steps, and pop/return the next batch of actions in the sequence.
        """
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
        if self.ensembled_actions is None:
            # Initializes `self._ensembled_action` to the sequence of actions predicted during the first
            # time step of the episode.
            self.ensembled_actions = actions.clone()
            # Note: The last dimension is unsqueeze to make sure we can broadcast properly for tensor
            # operations later.
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
            )
        else:
            # self.ensembled_actions will have shape (batch_size, chunk_size - 1, action_dim). Compute
            # the online update for those entries.
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            # The last action, which has no prior online average, needs to get concatenated onto the end.
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])]
            )
        # "Consume" the first action.
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action


class ACT(nn.Module):
    """Action Chunking Transformer: The underlying neural network for ACTPolicy.

    Note: In this code we use the terms `vae_encoder`, 'encoder', `decoder`. The meanings are as follows.
        - The `vae_encoder` is, as per the literature around variational auto-encoders (VAE), the part of the
          model that encodes the target data (a sequence of actions), and the condition (the robot
          joint-space).
        - A transformer with an `encoder` (not the VAE encoder) and `decoder` (not the VAE decoder) with
          cross-attention is used as the VAE decoder. For these terms, we drop the `vae_` prefix because we
          have an option to train this model without the variational objective (in which case we drop the
          `vae_encoder` altogether, and nothing about this model has anything to do with a VAE).
     注：本代码中使用术语`vae_encoder`、‘编码器’、`decoder`，其含义如下：
        - `vae_encoder`遵循变分自编码器（VAE）相关文献定义，指模型中负责编码目标数据（动作序列）与条件（机器人关节空间）的部分。       
        - 采用带交叉注意机制的Transformer模型作为VAE解码器，其编码器（非VAE编码器）与解码器（非VAE解码器）分别独立实现。
          对于这些术语，我们省略了`vae_`前缀，因为该模型可选择性地不采用变分目标进行训练
         （此时将完全移除`vae_encoder`，且该模型与VAE毫无关联）。

                                 Transformer
                                 Used alone for inference
                                 (acts as VAE decoder
                                  during training)
                                ┌───────────────────────┐
                                │             Outputs   │
                                │                ▲      │
                                │     ┌─────►┌───────┐  │
                   ┌──────┐     │     │      │Transf.│  │
                   │      │     │     ├─────►│decoder│  │
              ┌────┴────┐ │     │     │      │       │  │
              │         │ │     │ ┌───┴───┬─►│       │  │
              │ VAE     │ │     │ │       │  └───────┘  │
              │ encoder │ │     │ │Transf.│             │
              │         │ │     │ │encoder│             │
              └───▲─────┘ │     │ │       │             │
                  │       │     │ └▲──▲─▲─┘             │
                  │       │     │  │  │ │               │
                inputs    └─────┼──┘  │ image emb.      │
                                │    state emb.         │
                                └───────────────────────┘
    """
    
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] = None

    def __init__(self, config: ACTConfig):
        # BERT style VAE encoder with input tokens [cls, robot_state, *action_sequence].
        # The cls token forms parameters of the latent's distribution (like this [*means, *log_variances]).
        super().__init__()
        self.config = config
        self.enable_debug_visualization = False
        self.latest_yolo_debug_overlays: dict[str, np.ndarray] = {}
        self.pending_yolo_debug_overlays: dict[str, np.ndarray] = {}
        self.debug_yolo_call_count = 0
        self.latest_mask_weight_debug: dict[str, float | str | bool] = {}
        self.latest_mask_weight_masks: dict[str, Tensor] = {}
        self.latest_mask_weight_instance_masks: dict[str, Tensor] = {}
        self.latest_mask_weight_consistency: dict[str, dict[str, Tensor]] = {}
        self.latest_latent_sample: Tensor | None = None

        print('------customACT------') # customACT标记

        if self.config.use_vae:
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            # Projection layer for joint-space configuration to hidden dimension.
            if self.config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    self.config.robot_state_feature.shape[0], config.dim_model
                )
            # Projection layer for action (joint-space target) to hidden dimension.
            self.vae_encoder_action_input_proj = nn.Linear(
                self.config.action_feature.shape[0],
                config.dim_model,
            )
            # Projection layer from the VAE encoder's output to the latent distribution's parameter space.
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            # Fixed sinusoidal positional embedding for the input to the VAE encoder. Unsqueeze for batch
            # dimension.
            num_input_token_encoder = 1 + config.chunk_size
            if self.config.robot_state_feature:
                num_input_token_encoder += 1
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
            )




        # Backbone for image feature extraction.
        #修改为DinoV2Backbone
        if self.config.image_features:
            if self.config.vision_backbone == "dino":
                self.backbone = DinoV2Backbone()
            elif self.config.vision_backbone == "convnext":
                self.backbone = ConvNeXtBackbone()
            else:
                backbone_model = getattr(torchvision.models, config.vision_backbone)(
                    replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                    weights=config.pretrained_backbone_weights,
                    norm_layer=FrozenBatchNorm2d,
                )
                # Note: The assumption here is that we are using a ResNet model (and hence layer4 is the final
                # feature map).
                # Note: The forward method of this returns a dict: {"feature_map": output}.
                self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        if self.config.image_features:
            if config.vision_backbone == "dino":
                self.encoder_img_feat_input_proj = nn.Conv2d(
                    512, config.dim_model, kernel_size=1
                )
            elif config.vision_backbone == "convnext":
                self.encoder_img_feat_input_proj = nn.Conv2d(
                    512, config.dim_model, kernel_size=1
                )
            else:
                self.encoder_img_feat_input_proj = nn.Conv2d(
                    backbone_model.fc.in_features, config.dim_model, kernel_size=1
                )

        if self.config.image_features and self.config.use_mask_weight:
            self.mask_guided_visual_adapter = MaskGuidedVisualAdapter(config.dim_model, config.mw_config)
        



        # Transformer (acts as VAE decoder when training with the variational objective).
        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)

        # Transformer encoder input projections. The tokens will be structured like
        # [latent, (robot_state), (env_state), (image_feature_map_pixels)].
        if self.config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                self.config.robot_state_feature.shape[0], config.dim_model
            )
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                self.config.env_state_feature.shape[0], config.dim_model
            )

        # 新增：历史动作embedding模块
        if self.config.n_history_obs_states > 0:
            self.history_obs_state_embedding = HistoryObsStateEmbedding(config)
        
        # 如果会用到yolo
        if self.config.use_yolo:
            self.yolo_data_processer = YoloDataProcessor(config.seg_config, config.device)
        # 新增：实例分割理解模块
        if self.config.use_segment_understanding:
            # 初始化必需的插件
            self.kinematics = SimpleKinematics(config.seg_config.urdf_path, config.seg_config.ee_frame_name)
            # 动态赋值config
            yolo_nc = self.yolo_data_processer.yolo.nc
            config.seg_config.num_classes = yolo_nc
            config.seg_config.output_dim = config.dim_model
             # 初始化模块
            self.segment_understanding_embedding = SegmentUnderstandingEmbedding(config.seg_config)
        

        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        

        # Transformer encoder positional embeddings.
        n_1d_tokens = 1  # for the latent
        if self.config.robot_state_feature:
            n_1d_tokens += 1

        if self.config.env_state_feature:
            n_1d_tokens += 1

        if self.config.n_history_obs_states > 0:# 历史动作token
            n_1d_tokens += 4

        if self.config.use_segment_understanding:# 实例分割理解 token
            n_1d_tokens += 1


        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if self.config.image_features:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        # Transformer decoder.
        # Learnable positional embedding for the transformer's decoder (in the style of DETR object queries).
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        # Final action regression head on the output of the transformer's decoder.
        self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])

        self._reset_parameters()



        # 打印模型结构
        print("\n========== ACT MODEL STRUCT ==========")
        print(self.backbone)
        print("=====================================\n")

    @staticmethod
    def _count_yolo_detections(yolo_results) -> int:
        if yolo_results is None:
            return 0
        return sum(0 if result.boxes is None else len(result.boxes) for result in yolo_results)

    def _record_yolo_debug_from_results(
        self,
        img_key: str,
        imgs_for_yolo: Tensor,
        yolo_results,
        sources: tuple[str, ...],
        overlay_interval: int = 1,
    ) -> None:
        """Record Rerun debug data from the exact YOLO result consumed by policy branches."""
        if not self.enable_debug_visualization or yolo_results is None:
            return

        sources = tuple(dict.fromkeys(sources)) or ("yolo",)
        self.debug_yolo_call_count += 1
        cam_name = img_key.replace(f"{OBS_IMAGES}.", "")
        num_detections = self._count_yolo_detections(yolo_results)
        for source in sources:
            debug_prefix = f"custom.yolo_debug.{source}.{cam_name}"
            call_count = np.array(
                self.debug_yolo_call_count,
                dtype=np.float32,
            )
            num_detections_value = np.array(
                num_detections,
                dtype=np.float32,
            )
            call_count_key = f"{debug_prefix}.call_count"
            num_detections_key = f"{debug_prefix}.num_detections"
            self.latest_yolo_debug_overlays[call_count_key] = call_count
            self.latest_yolo_debug_overlays[num_detections_key] = num_detections_value
            self.pending_yolo_debug_overlays[call_count_key] = call_count
            self.pending_yolo_debug_overlays[num_detections_key] = num_detections_value

        interval = max(1, int(overlay_interval))
        if self.debug_yolo_call_count % interval != 0 or len(yolo_results) == 0:
            return

        overlay = self.yolo_data_processer.make_debug_overlay_chw(imgs_for_yolo[0], yolo_results[0])
        overlay_keys = [f"custom.yolo_overlay.{source}.{cam_name}" for source in sources]
        overlay_keys.append(f"custom.yolo_overlay.{cam_name}")
        for key in overlay_keys:
            self.latest_yolo_debug_overlays[key] = overlay
            self.pending_yolo_debug_overlays[key] = overlay


    def _reset_parameters(self):
        """Xavier-uniform initialization of the transformer parameters as in the original code."""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

                
        """A forward pass through the Action Chunking Transformer (with optional VAE encoder).

        `batch` should have the following structure:
        {
            [robot_state_feature] (optional): (B, state_dim) batch of robot states.

            [image_features]: (B, n_cameras, C, H, W) batch of images.
                AND/OR
            [env_state_feature]: (B, env_dim) batch of environment states.

            [action_feature] (optional, only if training with VAE): (B, chunk_size, action dim) batch of actions.
        }

        Returns:
            (B, chunk_size, action_dim) batch of action sequences
            Tuple containing the latent PDF's parameters (mean, log(σ²)) both as (B, L) tensors where L is the
            latent dimension.
        """
    # 实际执行动作预测的位置
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:

        forced_latent_sample = batch.get(_ACT_FORCED_LATENT_SAMPLE, None)
        if self.config.use_vae and self.training and forced_latent_sample is None:
            assert ACTION in batch, (
                "actions must be provided when using the variational objective in training mode."
            )
        batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]
        if self.enable_debug_visualization:
            self.latest_yolo_debug_overlays = {}
        self.latest_mask_weight_debug = {}
        self.latest_mask_weight_masks = {}
        self.latest_mask_weight_instance_masks = {}
        self.latest_mask_weight_consistency = {}

        # Prepare the latent for input to the transformer encoder.
        if forced_latent_sample is not None:
            mu = log_sigma_x2 = None
            latent_sample = forced_latent_sample.to(device=batch[OBS_STATE].device, dtype=batch[OBS_STATE].dtype)
        elif self.config.use_vae and ACTION in batch and self.training:
            # Prepare the input to the VAE encoder: [cls, *joint_space_configuration, *action_sequence].
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )  # (B, 1, D),为每个样本复制一个 class token embedding，作为序列的第 0 个 token。
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE])
                robot_state_embed = robot_state_embed.unsqueeze(1)  # (B, 1, D)
            action_embed = self.vae_encoder_action_input_proj(batch[ACTION])  # (B, S, D)

            if self.config.robot_state_feature:
                vae_encoder_input = [cls_embed, robot_state_embed, action_embed]  # (B, S+2, D)
            else:
                vae_encoder_input = [cls_embed, action_embed]
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

            # Prepare fixed positional embedding.
            # Note: detach() shouldn't be necessary but leaving it the same as the original code just in case.
            pos_embed = self.vae_encoder_pos_enc.clone().detach()  # (1, S+2, D)

            # Prepare key padding mask for the transformer encoder. We have 1 or 2 extra tokens at the start of the
            # sequence depending whether we use the input states or not (cls and robot state)
            # False means not a padding token.
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=batch[OBS_STATE].device,
            )
            key_padding_mask = torch.cat(
                [cls_joint_is_pad, batch["action_is_pad"]], axis=1
            )  # (bs, seq+1 or 2)

            # Forward pass through VAE encoder to get the latent PDF parameters.
            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]  # select the class token, with shape (B, D)
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            # This is 2log(sigma). Done this way to match the original implementation.
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]
            # Sample the latent with the reparameterization trick.
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            # When not using the VAE encoder, we set the latent to be all zeros.
            mu = log_sigma_x2 = None
            # TODO(rcadene, alexander-soare): remove call to `.to` to speedup forward ; precompute and use buffer
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
                batch[OBS_STATE].device
            )
        self.latest_latent_sample = latent_sample.detach()

        # Prepare transformer encoder inputs.
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        # Robot state token.
        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        # Environment state token.
        if self.config.env_state_feature:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))


        # 新增：调用历史观测状态
        if self.config.n_history_obs_states > 0:
            history_obs_state_embed = self.history_obs_state_embedding(batch[HIS_OBS_STATES])  # (B, D)
            # print(f"history_obs_state_embed: {history_obs_state_embed.shape}") # debug 输出历史观测状态embedding的形状
            # print(f"encoder_in_tokens length before adding history_obs_state_embed: {len(encoder_in_tokens)}") # debug 输出添加历史观测状态embedding前encoder_in_tokens的长度:2
            history_obs_state_embed = history_obs_state_embed.permute(1, 0, 2)  # (N, B, D)
            encoder_in_tokens.extend(list(history_obs_state_embed))
            # print(f"encoder_in_tokens length after adding history_obs_state_embed: {len(encoder_in_tokens)}") # debug 输出添加历史观测状态embedding后encoder_in_tokens的长度:6

        # 新增：调用实例分割理解模块
        yolo_results_by_img_key = {}
        imgs_for_yolo_by_img_key = {}
        forced_mask_weight_masks = batch.get(_MASK_WEIGHT_FORCED_MASKS, None)
        forced_mask_weight_instance_masks = batch.get(_MASK_WEIGHT_FORCED_INSTANCE_MASKS, None)
        skip_mask_weight_debug = bool(batch.get(_MASK_WEIGHT_SKIP_DEBUG, False)) or not bool(
            getattr(self.config.mw_config, "record_debug_log", True)
        )

        if self.config.use_segment_understanding:
            cam_key = f"observation.images.{self.config.seg_config.camera_name}"

            # 反归一化到 [0, 1] 范围，因为act的数据经过了mean-std归一化，不符合YOLO与FK输入需求
            norm_step:NormalizerProcessorStep = self.preprocessor.steps[3]
            imgs_for_yolo = norm_step._apply_transform(batch[cam_key], cam_key, FeatureType.VISUAL, inverse=True)
            obs_state_rad = norm_step._apply_transform(batch[OBS_STATE], OBS_STATE, FeatureType.STATE, inverse=True) * (torch.pi / 180.0) # 1、反归一化 2、度转弧度
            
            # 传入YOLO和FK，并得到分割理解的embedding
            yolo_results = self.yolo_data_processer.get_yolo_results(imgs_for_yolo)
            yolo_results_by_img_key[cam_key] = yolo_results
            imgs_for_yolo_by_img_key[cam_key] = imgs_for_yolo
            self._record_yolo_debug_from_results(
                cam_key,
                imgs_for_yolo,
                yolo_results,
                ("segment",),
                overlay_interval=1,
            )

            yolo_r_list = []
            yolo_mask_list = []
            for result in yolo_results:
                yolo_r_i, yolo_mask_i = self.yolo_data_processer.process_single_result(result)
                yolo_r_list.append(yolo_r_i)
                yolo_mask_list.append(yolo_mask_i)
            yolo_r = torch.stack(yolo_r_list, dim=0)
            yolo_mask = torch.stack(yolo_mask_list, dim=0)
            ee_pose = self.kinematics.forward_kinematics_batch(obs_state_rad)
            segment_understanding_embed = self.segment_understanding_embedding(yolo_r, yolo_mask , ee_pose) #(B, D)
            # 最后把embedding放入tokens
            encoder_in_tokens.append(segment_understanding_embed)
        

        if self.config.image_features:
            # For a list of images, the H and W may vary but H*W is constant.
            # NOTE: If modifying this section, verify on MPS devices that
            # gradients remain stable (no explosions or NaNs).
            # print(list(batch.keys()))

            img_keys = [k for k in batch.keys() if k.startswith(OBS_IMAGES + ".")]
            for img_key in img_keys:
                img = batch[img_key]    # [8, 3, 480, 640]
                cam_features = self.backbone(img)["feature_map"]    # [8, 3, 480, 640] -> [8, 512, 15, 20]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                yolo_mask = None
                yolo_instance_masks = None
                yolo_results = None
                imgs_for_yolo = None
                target_tokens = None
                target_pos_embed = None
                mask_geometry_token = None
                mask_geometry_pos_embed = None

                if self.config.use_mask_weight:
                    forced_yolo_mask = (
                        forced_mask_weight_masks.get(img_key)
                        if isinstance(forced_mask_weight_masks, dict)
                        else None
                    )
                    forced_yolo_instance_masks = (
                        forced_mask_weight_instance_masks.get(img_key)
                        if isinstance(forced_mask_weight_instance_masks, dict)
                        else None
                    )
                    if forced_yolo_mask is not None:
                        yolo_mask = forced_yolo_mask.to(device=img.device)
                        if forced_yolo_instance_masks is not None:
                            yolo_instance_masks = forced_yolo_instance_masks.to(device=img.device)
                    else:
                        if img_key in yolo_results_by_img_key:
                            imgs_for_yolo = imgs_for_yolo_by_img_key[img_key]
                            yolo_results = yolo_results_by_img_key[img_key]
                        else:
                            norm_step:NormalizerProcessorStep = self.preprocessor.steps[3]
                            imgs_for_yolo = norm_step._apply_transform(img, img_key, FeatureType.VISUAL, inverse=True)
                            yolo_results = self.yolo_data_processer.get_yolo_results(imgs_for_yolo)
                            yolo_results_by_img_key[img_key] = yolo_results
                            imgs_for_yolo_by_img_key[img_key] = imgs_for_yolo
                        self._record_yolo_debug_from_results(
                            img_key,
                            imgs_for_yolo,
                            yolo_results,
                            ("mask_weight",),
                            overlay_interval=1,
                        )
                        yolo_mask = yolo_result_to_soft_mask(
                            yolo_results,
                            kernel_size=getattr(self.config.mw_config, "mask_blur_kernel_size", 7),
                            sigma=getattr(self.config.mw_config, "mask_blur_sigma", 2.0),
                        )
                        if getattr(self.config.mw_config, "use_instance_object_tokens", True):
                            yolo_instance_masks, _, _ = yolo_results_to_instance_masks(
                                yolo_results,
                                max_instances=getattr(
                                    self.config.mw_config,
                                    "max_instance_object_tokens",
                                    4,
                                ),
                                kernel_size=getattr(self.config.mw_config, "mask_blur_kernel_size", 7),
                                sigma=getattr(self.config.mw_config, "mask_blur_sigma", 2.0),
                            )
                    self.latest_mask_weight_masks[img_key] = yolo_mask.detach()
                    if yolo_instance_masks is not None:
                        self.latest_mask_weight_instance_masks[img_key] = yolo_instance_masks.detach()

                    if not skip_mask_weight_debug:
                        cam_name = img_key.replace(f"{OBS_IMAGES}.", "")
                        mask_for_debug = yolo_mask.detach()
                        self.latest_mask_weight_debug[f"mask_weight/{cam_name}/detections"] = float(
                            self._count_yolo_detections(yolo_results)
                        )
                        self.latest_mask_weight_debug[f"mask_weight/{cam_name}/mask_mean"] = float(
                            mask_for_debug.mean().item()
                        )
                        self.latest_mask_weight_debug[f"mask_weight/{cam_name}/mask_max"] = float(
                            mask_for_debug.max().item()
                        )
                        self.latest_mask_weight_debug[f"mask_weight/{cam_name}/mask_coverage"] = float(
                            (mask_for_debug > 0.05).float().mean().item()
                        )
                        if yolo_instance_masks is not None:
                            instance_for_debug = yolo_instance_masks.detach()
                            self.latest_mask_weight_debug[f"mask_weight/{cam_name}/instance_masks"] = float(
                                instance_for_debug.shape[1]
                            )
                            self.latest_mask_weight_debug[f"mask_weight/{cam_name}/instance_mask_valid_count"] = float(
                                (instance_for_debug.amax(dim=(2, 3, 4)) > 0.0).float().sum(dim=1).mean().item()
                            )

                # 投影features到指定维度
                cam_features = self.encoder_img_feat_input_proj(cam_features)    # [8, 512, 15, 20] -> [8, 512, 15, 20]

                if self.config.use_mask_weight:
                    target_h, target_w = cam_features.shape[-2:]
                    mask_resized = torch.nn.functional.interpolate(
                        yolo_mask,
                        size=(target_h, target_w),
                        mode='bilinear',
                        align_corners=False
                    )
                    mask_resized = mask_resized.to(dtype=cam_features.dtype, device=cam_features.device)
                    mask_resized = torch.clamp(mask_resized, 0.0, 1.0)
                    (
                        cam_features,
                        target_tokens,
                        target_pos_embed,
                        mask_geometry_token,
                        mask_geometry_pos_embed,
                    ) = self.mask_guided_visual_adapter(
                        cam_features,
                        mask_resized,
                        instance_masks=yolo_instance_masks,
                        record_debug=not skip_mask_weight_debug,
                    )
                    self.latest_mask_weight_consistency[img_key] = dict(
                        getattr(self.mask_guided_visual_adapter, "latest_consistency", {}) or {}
                    )
                    if not skip_mask_weight_debug:
                        cam_name = img_key.replace(f"{OBS_IMAGES}.", "")
                        gate_scale = getattr(self.mask_guided_visual_adapter, "gate_scale", None)
                        if gate_scale is not None:
                            self.latest_mask_weight_debug[f"mask_weight/{cam_name}/gate_scale"] = float(
                                gate_scale.detach().item()
                            )
                        adapter_debug = getattr(self.mask_guided_visual_adapter, "latest_debug", {})
                        for debug_name, debug_value in adapter_debug.items():
                            self.latest_mask_weight_debug[
                                f"mask_weight/{cam_name}/adapter_{debug_name}"
                            ] = float(debug_value)
                        if target_tokens is not None:
                            self.latest_mask_weight_debug[f"mask_weight/{cam_name}/target_tokens"] = float(
                                target_tokens.shape[0]
                            )
                        if mask_geometry_token is not None:
                            self.latest_mask_weight_debug[f"mask_weight/{cam_name}/mask_geometry_tokens"] = float(
                                mask_geometry_token.shape[0]
                            )


                # 重新排列features形状
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                # Extend immediately instead of accumulating and concatenating
                # Convert to list to extend properly
                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))
                if target_tokens is not None:
                    encoder_in_tokens.extend(list(target_tokens))
                    encoder_in_pos_embed.extend(list(target_pos_embed))
                if mask_geometry_token is not None:
                    encoder_in_tokens.extend(list(mask_geometry_token))
                    encoder_in_pos_embed.extend(list(mask_geometry_pos_embed))

        # Stack all tokens along the sequence dimension.
        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        # Forward pass through the transformer modules.
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        # TODO(rcadene, alexander-soare): remove call to `device` ; precompute and use buffer
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        # Move back to (B, S, C).
        decoder_out = decoder_out.transpose(0, 1)

        actions = self.action_head(decoder_out)

        return actions, (mu, log_sigma_x2)


class ACTEncoder(nn.Module):
    """Convenience module for running multiple encoder layers, maybe followed by normalization."""

    def __init__(self, config: ACTConfig, is_vae_encoder: bool = False):
        super().__init__()
        self.is_vae_encoder = is_vae_encoder
        num_layers = config.n_vae_encoder_layers if self.is_vae_encoder else config.n_encoder_layers
        self.layers = nn.ModuleList([ACTEncoderLayer(config) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(
        self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x


class ACTEncoderLayer(nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def forward(self, x, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)
        x = x[0]  # note: [0] to select just the output, not the attention weights
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x


class ACTDecoder(nn.Module):
    def __init__(self, config: ACTConfig):
        """Convenience module for running multiple decoder layers followed by normalization."""
        super().__init__()
        self.layers = nn.ModuleList([ACTDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(
                x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed
            )
        if self.norm is not None:
            x = self.norm(x)
        return x


class ACTDecoderLayer(nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.multihead_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: (Decoder Sequence, Batch, Channel) tensor of input tokens.
            encoder_out: (Encoder Sequence, B, C) output features from the last layer of the encoder we are
                cross-attending with.
            encoder_pos_embed: (ES, 1, C) positional embedding for keys (from the encoder).
            decoder_pos_embed: (DS, 1, C) positional embedding for the queries (from the decoder).
        Returns:
            (DS, B, C) tensor of decoder output features.
        """
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]  # select just the output, not the attention weights
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]  # select just the output, not the attention weights
        x = skip + self.dropout2(x)
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """1D sinusoidal positional embeddings as in Attention is All You Need.

    Args:
        num_positions: Number of token positions required.
    Returns: (num_positions, dimension) position embeddings (the first dimension is the batch dimension).

    """

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.from_numpy(sinusoid_table).float()


class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embeddings similar to what's presented in Attention Is All You Need.

    The variation is that the position indices are normalized in [0, 2π] (not quite: the lower bound is 1/H
    for the vertical direction, and 1/W for the horizontal direction.
    """

    def __init__(self, dimension: int):
        """
        Args:
            dimension: The desired dimension of the embeddings.
        """
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        # Inverse "common ratio" for the geometric progression in sinusoid frequencies.
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: A (B, C, H, W) batch of 2D feature map to generate the embeddings for.
        Returns:
            A (1, C, H, W) batch of corresponding sinusoidal positional embeddings.
        """
        not_mask = torch.ones_like(x[0, :1])  # (1, H, W)
        # Note: These are like range(1, H+1) and range(1, W+1) respectively, but in most implementations
        # they would be range(0, H) and range(0, W). Keeping it at as is to match the original code.
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        # "Normalize" the position index such that it ranges in [0, 2π].
        # Note: Adding epsilon on the denominator should not be needed as all values of y_embed and x_range
        # are non-zero by construction. This is an artifact of the original code.
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )

        x_range = x_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)
        y_range = y_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)

        # Note: this stack then flatten operation results in interleaved sine and cosine terms.
        # pos_embed_x and pos_embed_y are (1, H, W, C // 2).
        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)  # (1, C, H, W)

        return pos_embed


def get_activation_fn(activation: str) -> Callable:
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
