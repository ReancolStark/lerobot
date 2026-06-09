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

import logging
import math
from collections import deque
from collections.abc import Callable
from itertools import chain

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from safetensors.torch import load_file as load_safetensor_file
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d
from lerobot.policies.dino_act.dino_backbone import DinoV2Backbone

from lerobot.policies.customACT.configuration_customACT import ACTConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import log_model_loading_keys
from lerobot.utils.constants import (
    ACTION,
    ACTION_HISTORY,
    HISTORY_MASK,
    HIS_OBS_STATES,
    OBS_ENV_STATE,
    OBS_IMAGES,
    OBS_STATE,
    OBS_STATE_HISTORY,
)
#新增：自己的import
from lerobot.policies.customACT.history_obs_state.modeling_history_obs import HistoryObsStateEmbedding
from lerobot.policies.customACT.replan_score_adaptive_chunking import (
    HistoryTokenReplanScoreModel,
    ReplanScoreAdaptiveChunkingController,
    ThreeRegimeAdaptiveChunkingController,
    compute_replan_score_loss,
)
from lerobot.policies.customACT.key_history_state.modeling_key_history import KeyHistoryTokenEncoder
from lerobot.policies.dino_act.backbone_res import ResNet18Backbone, get_custom_backbone
from lerobot.policies.dino_act.convnext import ConvNeXtBackbone
from lerobot.policies.dino_act.convnext_frame import ConvNeXtBackbone1
from lerobot.policies.customACT.segment_understanding.modeling_segment_understanding import SegmentUnderstandingEmbedding
from lerobot.policies.customACT.segment_understanding.utils.kinematics import SimpleKinematics
from lerobot.policies.customACT.segment_understanding.utils.yolo_data_processer import YoloDataProcessor
# from lerobot.policies.customACT.segment_understanding.utils.denormalize import denormalize_img_with_mean_stats,denormalize_obs_and_angle_to_rad
# 由于想要未经初始化的数据，需要用到这个
from lerobot.processor import PolicyProcessorPipeline
from typing import Any
from lerobot.processor.normalize_processor import NormalizerProcessorStep
from lerobot.configs.types import FeatureType

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

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.three_regime_adaptive_chunker = None
        if config.use_adaptive_action_chunking:
            self.three_regime_adaptive_chunker = ThreeRegimeAdaptiveChunkingController(
                config.history_token_replan_score,
                policy_chunk_size=config.chunk_size,
                policy_n_action_steps=(
                    config.chunk_size if config.temporal_ensemble_coeff is not None else config.n_action_steps
                ),
            )

        self.replan_score_adaptive_chunker = None
        if (
            config.use_history_token_adaptive_chunking
            and config.history_token_adaptive_chunking.mode == "replan_score"
        ):
            self.replan_score_adaptive_chunker = ReplanScoreAdaptiveChunkingController(
                config.history_token_adaptive_chunking,
                policy_chunk_size=config.chunk_size,
                policy_n_action_steps=config.n_action_steps,
            )

        self.reset()

    @classmethod
    def _load_as_safetensor(cls, model: "ACTPolicy", model_file: str, map_location: str, strict: bool) -> "ACTPolicy":
        state_dict = load_safetensor_file(model_file, device=map_location)
        state_dict = cls._remap_legacy_history_token_state_dict_keys(model, state_dict)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
        log_model_loading_keys(missing_keys, unexpected_keys)
        return model

    @staticmethod
    def _remap_legacy_history_token_state_dict_keys(
        model: "ACTPolicy",
        state_dict: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Load history-token checkpoints saved before the module rename.

        Older experiments saved the same history encoder under
        ``model.recovery_history_encoder`` with a standalone
        ``model.recovery_token_pos_embed`` parameter, and later under
        ``model.recovery_adaptive_chunking_model``. The current code calls the
        branch ``model.history_token_replan_score_model``. Without this remap
        the replan-score head is silently left randomly initialized when
        ``strict=False`` is used for inference.
        """
        target_state = model.state_dict()
        remapped: dict[str, Tensor] = {}
        remapped_count = 0

        for key, value in state_dict.items():
            new_key = key
            if key.startswith("model.recovery_history_encoder."):
                new_key = key.replace(
                    "model.recovery_history_encoder.",
                    "model.history_token_replan_score_model.",
                    1,
                )
            elif key.startswith("model.recovery_adaptive_chunking_model."):
                new_key = key.replace(
                    "model.recovery_adaptive_chunking_model.",
                    "model.history_token_replan_score_model.",
                    1,
                )
            elif key == "model.recovery_token_pos_embed":
                new_key = "model.history_token_replan_score_model.token_pos_embed"

            if ".recovery_score_head." in new_key:
                new_key = new_key.replace(".recovery_score_head.", ".replan_score_head.")

            if new_key != key and new_key in target_state:
                if tuple(target_state[new_key].shape) == tuple(value.shape):
                    remapped[new_key] = value
                    remapped_count += 1
                    continue
                logging.warning(
                    "Skipping legacy history-token checkpoint key remap %s -> %s due to shape mismatch: %s vs %s",
                    key,
                    new_key,
                    tuple(value.shape),
                    tuple(target_state[new_key].shape),
                )

            remapped[key] = value

        if remapped_count:
            logging.info("Remapped %s legacy history-token checkpoint key(s).", remapped_count)
        return remapped

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
        if self.three_regime_adaptive_chunker is not None:
            self.three_regime_adaptive_chunker.reset()
        if self.replan_score_adaptive_chunker is not None:
            self.replan_score_adaptive_chunker.reset()
        if self.config.use_history_token_replan_score:
            self._history_state_buffer = deque([], maxlen=self.config.history_len)
            self._history_action_buffer = deque([], maxlen=self.config.history_len)
        if self.config.use_key_history_token:
            self._key_history_state_buffer = deque([], maxlen=self.config.key_history_len)

    def _record_history_token_state(self, batch: dict[str, Tensor]) -> None:
        """Record the current normalized proprioceptive state for online inference.

        During training, the dataloader provides ``observation.state.history`` directly.
        During real robot rollout, there is no dataloader window, so the policy keeps a
        small FIFO buffer here. The stored tensor is [state_dim] and is already normalized
        by the policy preprocessor, matching the training-time state history scale.
        """
        if not self.config.use_history_token_replan_score or OBS_STATE_HISTORY in batch:
            return
        state = batch[OBS_STATE].detach()
        if state.ndim == 1:
            state = state.unsqueeze(0)
        assert state.shape[0] == 1, "Online history-token buffer currently supports batch size 1."
        self._history_state_buffer.append(state[0])

    def _record_key_history_state(self, batch: dict[str, Tensor]) -> None:
        """Record current normalized state for online key-history inference."""
        if not self.config.use_key_history_token or OBS_STATE_HISTORY in batch:
            return
        state = batch[OBS_STATE].detach()
        if state.ndim == 1:
            state = state.unsqueeze(0)
        assert state.shape[0] == 1, "Online key history buffer currently supports batch size 1."
        self._key_history_state_buffer.append(state[0])

    def _record_history_token_action(self, action: Tensor) -> None:
        """Record the normalized action selected at this control step.

        The history-token branch expects ``action.history = [a_{t-H}, ..., a_{t-1}]``.
        We therefore append the action only after it has been selected from the chunk or
        temporal ensemble. This keeps the next model call causal: it can see actions that
        were already issued, but never future actions from the current predicted chunk.
        """
        if not self.config.use_history_token_replan_score:
            return
        action = action.detach()
        if action.ndim == 1:
            action = action.unsqueeze(0)
        assert action.shape[0] == 1, "Online history-token action buffer currently supports batch size 1."
        self._history_action_buffer.append(action[0])

    def _left_pad_history(
        self,
        values: list[Tensor],
        *,
        target_len: int,
        pad_value: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Convert a variable-length FIFO buffer into a fixed history tensor.

        Args:
            values: Recent tensors, each with shape [D]. Newer elements are at the end.
            target_len: Desired history length H.
            pad_value: [1, D] tensor used for missing early-episode positions.

        Returns:
            padded: [H, D], left padded so the newest value is at index H - 1.
            mask: [H], True only where the entry came from real rollout history.
        """
        values = values[-target_len:]
        valid_len = len(values)
        if valid_len == 0:
            padded = pad_value.repeat(target_len, 1)
        else:
            pad = pad_value.repeat(target_len - valid_len, 1)
            padded = torch.cat([pad, torch.stack(values, dim=0)], dim=0)
        mask = torch.zeros(target_len, dtype=torch.bool, device=pad_value.device)
        if valid_len > 0:
            mask[-valid_len:] = True
        return padded, mask

    def _add_history_token_inputs_to_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Attach history-token fields to an online inference batch.

        Training batches get these fields from dataset delta timestamps. In deployment,
        this method builds:
          - observation.state.history: [1, H, state_dim]
          - action.history: [1, H, action_dim]
          - history_mask: [1, H]

        The mask requires both a real state and a real previous action at the same history
        position, so the first few control steps do not pretend that padded actions are
        valid evidence.
        """
        if not self.config.use_history_token_replan_score or OBS_STATE_HISTORY in batch:
            return batch
        if len(self._history_state_buffer) == 0:
            self._record_history_token_state(batch)

        batch = dict(batch)
        current_state = batch[OBS_STATE]
        if current_state.ndim == 1:
            current_state = current_state.unsqueeze(0)
        assert current_state.shape[0] == 1, "Online history-token buffer currently supports batch size 1."

        state_pad_value = self._history_state_buffer[0].view(1, -1).to(current_state.device)
        action_dim = self.config.action_feature.shape[0]
        action_pad_value = torch.zeros(1, action_dim, dtype=current_state.dtype, device=current_state.device)

        state_values = [v.to(current_state.device, dtype=current_state.dtype) for v in self._history_state_buffer]
        action_values = [v.to(current_state.device, dtype=current_state.dtype) for v in self._history_action_buffer]
        state_history, state_mask = self._left_pad_history(
            state_values, target_len=self.config.history_len, pad_value=state_pad_value
        )
        action_history, action_mask = self._left_pad_history(
            action_values, target_len=self.config.history_len, pad_value=action_pad_value
        )

        batch[OBS_STATE_HISTORY] = state_history.unsqueeze(0)
        batch[ACTION_HISTORY] = action_history.unsqueeze(0)
        batch[HISTORY_MASK] = (state_mask & action_mask).unsqueeze(0)
        return batch

    def _add_key_history_to_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Attach key-history state fields to an online inference batch.

        Training batches get observation.state.history from dataset delta timestamps.
        In deployment, this method builds:
          - observation.state.history: [1, H, state_dim]
          - history_mask: [1, H]
        """
        if not self.config.use_key_history_token or OBS_STATE_HISTORY in batch:
            return batch
        if len(self._key_history_state_buffer) == 0:
            self._record_key_history_state(batch)

        batch = dict(batch)
        current_state = batch[OBS_STATE]
        if current_state.ndim == 1:
            current_state = current_state.unsqueeze(0)
        assert current_state.shape[0] == 1, "Online key history buffer currently supports batch size 1."

        state_pad_value = self._key_history_state_buffer[0].view(1, -1).to(current_state.device)
        state_values = [
            v.to(current_state.device, dtype=current_state.dtype) for v in self._key_history_state_buffer
        ]
        state_history, state_mask = self._left_pad_history(
            state_values,
            target_len=self.config.key_history_len,
            pad_value=state_pad_value,
        )

        batch[OBS_STATE_HISTORY] = state_history.unsqueeze(0)
        batch[HISTORY_MASK] = state_mask.unsqueeze(0)
        return batch

    def _observe_adaptive_action_chunking_state(self, batch: dict[str, Tensor]) -> None:
        if self.three_regime_adaptive_chunker is None:
            return
        self.three_regime_adaptive_chunker.observe_state(batch.get(OBS_STATE))

    def prepare_online_inference_step(self, batch: dict[str, Tensor]) -> None:
        """Update online history buffers before a policy inference call."""
        self._record_history_token_state(batch)
        self._record_key_history_state(batch)
        self._observe_adaptive_action_chunking_state(batch)

    def _get_replan_score_for_adaptive_action_chunking(self) -> float | None:
        replan_aux_outputs = getattr(self.model, "replan_aux_outputs", None)
        if not replan_aux_outputs:
            return None
        replan_score = replan_aux_outputs.get("replan_score", replan_aux_outputs.get("recovery_score"))
        if replan_score is None:
            return None
        return float(replan_score.detach().mean().item())

    def _decide_three_regime_adaptive_action_chunking(self, actions: Tensor):
        if self.three_regime_adaptive_chunker is None:
            return None
        return self.three_regime_adaptive_chunker.decide(
            actions,
            replan_score=self._get_replan_score_for_adaptive_action_chunking(),
        )

    def _adapt_action_chunk_with_replan_score(
        self,
        actions: Tensor,
        *,
        max_actions_per_chunk: int | None = None,
    ) -> Tensor:
        if self.replan_score_adaptive_chunker is None:
            raise RuntimeError("Replan-score adaptive chunker is not enabled.")

        decision = self.replan_score_adaptive_chunker.decide(
            replan_score=self._get_replan_score_for_adaptive_action_chunking(),
            available_actions=actions.shape[1],
            max_actions_per_chunk=max_actions_per_chunk,
        )
        selected_actions = actions[:, : decision.chunk_size]
        selected_actions = self.replan_score_adaptive_chunker.smooth_chunk_transition(
            selected_actions,
            decision=decision,
        )
        self.replan_score_adaptive_chunker.debug_prediction(
            predicted_actions=actions,
            executed_actions=selected_actions,
            decision=decision,
            source="replan_score_chunk",
        )
        return selected_actions

    def adapt_action_chunk_for_inference(
        self,
        actions: Tensor,
        *,
        max_actions_per_chunk: int | None = None,
    ) -> Tensor:
        """Return the action prefix selected by the adaptive chunk controller."""
        if self.replan_score_adaptive_chunker is not None:
            return self._adapt_action_chunk_with_replan_score(
                actions,
                max_actions_per_chunk=max_actions_per_chunk,
            )

        if self.three_regime_adaptive_chunker is None:
            if max_actions_per_chunk is None:
                max_actions_per_chunk = self.config.n_action_steps
            return actions[:, :max_actions_per_chunk]

        decision = self._decide_three_regime_adaptive_action_chunking(actions)
        chunk_size = decision.chunk_size
        if max_actions_per_chunk is not None:
            chunk_size = min(chunk_size, max_actions_per_chunk)
        selected_actions = actions[:, :chunk_size]
        selected_actions = self.three_regime_adaptive_chunker.smooth_chunk_transition(selected_actions)
        self.three_regime_adaptive_chunker.debug_prediction(
            predicted_actions=actions,
            executed_actions=selected_actions,
            decision=decision,
            source="action_chunk",
        )
        return selected_actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()  # keeping the policy in eval mode as it could be set to train mode while queue is consumed
        self.prepare_online_inference_step(batch)

        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            decision = self._decide_three_regime_adaptive_action_chunking(actions)
            if decision is not None:
                action = self.three_regime_adaptive_chunker.update_temporal_ensemble(
                    actions,
                    old_action_weight=decision.old_action_weight,
                )
                self.three_regime_adaptive_chunker.debug_prediction(
                    predicted_actions=actions,
                    executed_actions=actions[:, :1],
                    decision=decision,
                    source="temporal_ensemble",
                )
                remaining = (
                    0
                    if self.three_regime_adaptive_chunker.ensembled_actions is None
                    else self.three_regime_adaptive_chunker.ensembled_actions.shape[1]
                )
                self.three_regime_adaptive_chunker.debug_execution(
                    action=action,
                    remaining_actions=remaining,
                    source="temporal_ensemble",
                )
                self.three_regime_adaptive_chunker.observe_executed_action(action)
            else:
                action = self.temporal_ensembler.update(actions)
            self._record_history_token_action(action)
            return action

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)
            actions = self.adapt_action_chunk_for_inference(actions)

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))
        action = self._action_queue.popleft()
        if self.three_regime_adaptive_chunker is not None:
            self.three_regime_adaptive_chunker.debug_execution(
                action=action,
                remaining_actions=len(self._action_queue),
                source="select_action",
            )
            self.three_regime_adaptive_chunker.observe_executed_action(action)
        elif self.replan_score_adaptive_chunker is not None:
            self.replan_score_adaptive_chunker.debug_execution(
                action=action,
                remaining_actions=len(self._action_queue),
                source="select_action",
            )
            self.replan_score_adaptive_chunker.observe_executed_action(action)
        self._record_history_token_action(action)
        return action

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor: # 实际运行时的100步动作预测
        """Predict a chunk of actions given environment observations."""
        self.eval()

        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        batch = self._add_history_token_inputs_to_batch(batch)
        batch = self._add_key_history_to_batch(batch)

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

        history_aux_action_hat = getattr(self.model, "history_aux_action_hat", None)
        if (
            self.config.n_history_obs_states > 0
            and self.config.ho_aux_loss_weight > 0
            and history_aux_action_hat is not None
        ):
            first_action_mask = (~batch["action_is_pad"][:, 0]).unsqueeze(-1)
            history_aux_action_loss = (
                F.l1_loss(batch[ACTION][:, 0], history_aux_action_hat, reduction="none") * first_action_mask
            ).mean()
            loss = loss + self.config.ho_aux_loss_weight * history_aux_action_loss
            loss_dict["history_aux_action_loss"] = history_aux_action_loss.item()
            loss_dict["history_aux_loss_weight"] = self.config.ho_aux_loss_weight

        replan_aux_outputs = getattr(self.model, "replan_aux_outputs", None)
        if self.config.use_history_token_replan_score and replan_aux_outputs is not None:
            # Auxiliary target 1: from the history tokens alone, predict the first action
            # in the supervised ACT chunk, i.e. batch["action"][:, 0, :].
            first_action_mask = (~batch["action_is_pad"][:, 0]).unsqueeze(-1).to(dtype=batch[ACTION].dtype)
            hist_action_loss = (
                F.mse_loss(
                    replan_aux_outputs["hist_action_pred"],
                    batch[ACTION][:, 0],
                    reduction="none",
                )
                * first_action_mask
            ).mean()

            # Auxiliary target 2: keep the learned event score weakly aligned with a
            # no-label motion-change prior. This does not introduce manual phase labels;
            # it only says that large velocity/acceleration/execution-error moments are
            # plausible replan-relevant events.
            history_mask = batch[HISTORY_MASK].to(dtype=torch.bool)
            valid = history_mask.to(dtype=batch[ACTION].dtype)
            event_prior = replan_aux_outputs["event_prior"]
            event_scores = replan_aux_outputs["event_scores"]
            denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
            prior_mean = (event_prior * valid).sum(dim=1, keepdim=True) / denom
            prior_var = (((event_prior - prior_mean) * valid) ** 2).sum(dim=1, keepdim=True) / denom
            normalized_prior = (event_prior - prior_mean) / (prior_var.sqrt() + 1e-6)
            event_prior_loss = (
                (torch.sigmoid(event_scores) - torch.sigmoid(normalized_prior)).pow(2) * valid
            ).sum() / valid.sum().clamp_min(1.0)

            loss = (
                loss
                + self.config.history_token_replan_score.action_loss_weight * hist_action_loss
                + self.config.history_token_replan_score.event_prior_loss_weight * event_prior_loss
            )
            loss_dict["hist_action_loss"] = hist_action_loss.item()
            loss_dict["event_prior_loss"] = event_prior_loss.item()
            loss_dict["replan_score_mean"] = replan_aux_outputs["replan_score"].mean().item()
            if history_mask.any():
                loss_dict["event_score_mean"] = event_scores[history_mask].mean().item()
            else:
                loss_dict["event_score_mean"] = 0.0

            if self.config.history_token_replan_score.replan_score_loss_weight > 0:
                replan_score_loss, replan_target_info = compute_replan_score_loss(
                    replan_score=replan_aux_outputs["replan_score"],
                    state_history=batch[OBS_STATE_HISTORY],
                    action_history=batch[ACTION_HISTORY],
                    future_actions=batch[ACTION],
                    history_mask=batch[HISTORY_MASK],
                    action_is_pad=batch.get("action_is_pad"),
                    config=self.config.history_token_replan_score,
                )
                loss = (
                    loss
                    + self.config.history_token_replan_score.replan_score_loss_weight
                    * replan_score_loss
                )
                loss_dict["replan_score_loss"] = replan_score_loss.item()
                loss_dict["replan_target_mean"] = replan_target_info["target"].mean().item()
                loss_dict["replan_raw_score_mean"] = replan_target_info["raw_score"].mean().item()
                loss_dict["future_action_correction_mean"] = (
                    replan_target_info["future_action_correction"].mean().item()
                )

        key_history_aux_outputs = getattr(self.model, "key_history_aux_outputs", None)
        if self.config.use_key_history_token and key_history_aux_outputs is not None:
            first_action_mask = (~batch["action_is_pad"][:, 0]).unsqueeze(-1).to(dtype=batch[ACTION].dtype)

            if self.config.key_history_action_loss_weight > 0:
                key_hist_action_loss = (
                    F.mse_loss(
                        key_history_aux_outputs["hist_action_pred"],
                        batch[ACTION][:, 0],
                        reduction="none",
                    )
                    * first_action_mask
                ).mean()
                loss = loss + self.config.key_history_action_loss_weight * key_hist_action_loss
                loss_dict["key_history_action_loss"] = key_hist_action_loss.item()

            if self.config.key_history_event_loss_weight > 0:
                history_mask = batch[HISTORY_MASK].to(dtype=torch.bool)
                valid = history_mask.to(dtype=batch[ACTION].dtype)
                event_prior = key_history_aux_outputs["event_prior"]
                event_scores = key_history_aux_outputs["event_scores"]
                denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
                prior_mean = (event_prior * valid).sum(dim=1, keepdim=True) / denom
                prior_var = (((event_prior - prior_mean) * valid) ** 2).sum(dim=1, keepdim=True) / denom
                normalized_prior = (event_prior - prior_mean) / (prior_var.sqrt() + 1e-6)
                key_event_prior_loss = (
                    (torch.sigmoid(event_scores) - torch.sigmoid(normalized_prior)).pow(2) * valid
                ).sum() / valid.sum().clamp_min(1.0)
                loss = loss + self.config.key_history_event_loss_weight * key_event_prior_loss
                loss_dict["key_history_event_prior_loss"] = key_event_prior_loss.item()

            history_mask = batch[HISTORY_MASK].to(dtype=torch.bool)
            event_scores = key_history_aux_outputs["event_scores"]
            if history_mask.any():
                loss_dict["key_history_event_score_mean"] = event_scores[history_mask].mean().item()
            else:
                loss_dict["key_history_event_score_mean"] = 0.0

        return loss, loss_dict

    # 专门给yolo和fk用的预处理器设置函数，它们需要没有经过归一化的数据，然而lerobot传入的batch已经经过归一化了
    def set_preprocessor(self, preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,):
        self.model.preprocessor = preprocessor
        

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
            self.history_aux_action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])
            self.history_aux_action_hat = None

        history_replan_cfg = self.config.history_token_replan_score
        if self.config.use_history_token_replan_score:
            self.history_token_replan_score_model = HistoryTokenReplanScoreModel(
                state_dim=self.config.robot_state_feature.shape[0],
                action_dim=self.config.action_feature.shape[0],
                dim_model=config.dim_model,
                config=history_replan_cfg,
            )
            self.replan_aux_outputs = None

        if self.config.use_key_history_token:
            self.key_history_encoder = KeyHistoryTokenEncoder(
                state_dim=self.config.robot_state_feature.shape[0],
                action_dim=self.config.action_feature.shape[0],
                dim_model=config.dim_model,
                history_len=config.key_history_len,
                num_segments=config.key_history_num_segments,
                hidden_dim=config.key_history_hidden_dim,
                conv_kernel_size=config.key_history_conv_kernel_size,
                conv_dilations=config.key_history_conv_dilations,
                dropout=config.key_history_dropout,
                prior_scale_init=config.key_history_prior_scale_init,
                selection_temperature=config.key_history_selection_temperature,
            )
            self.key_history_token_pos_embed = nn.Parameter(
                torch.zeros(config.key_history_num_segments, config.dim_model)
            )
            self.key_history_aux_outputs = None
        







        # 新增：实例分割理解模块
        if self.config.use_segment_understanding:
            # 初始化两个必需的插件：FK & YOLO
            self.kinematics = SimpleKinematics(config.seg_config.urdf_path, config.seg_config.ee_frame_name)
            self.yolo_data_processer = YoloDataProcessor(config.seg_config, config.device)
            # 动态赋值config
            yolo_nc = self.yolo_data_processer.yolo.nc
            config.seg_config.num_classes = yolo_nc
            config.seg_config.output_dim = config.dim_model
             # 初始化模块
            self.segment_understanding_embedding = SegmentUnderstandingEmbedding(config.seg_config)






        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        




        # backbone
        

        








        # Transformer encoder positional embeddings.
        n_1d_tokens = 1  # for the latent
        if self.config.robot_state_feature:
            n_1d_tokens += 1

        if self.config.env_state_feature:
            n_1d_tokens += 1

        if self.config.n_history_obs_states > 0:# 历史动作token
            n_1d_tokens += self.config.ho_history_segment_num

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

        if self.config.use_history_token_replan_score:
            self.replan_aux_outputs = None
        if self.config.use_key_history_token:
            self.key_history_aux_outputs = None

        if self.config.use_vae and self.training:
            assert ACTION in batch, (
                "actions must be provided when using the variational objective in training mode."
            )
        batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]

        # Prepare the latent for input to the transformer encoder.
        if self.config.use_vae and ACTION in batch and self.training:
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
            self.history_aux_action_hat = self.history_aux_action_head(history_obs_state_embed.mean(dim=1))
            # print(f"history_obs_state_embed: {history_obs_state_embed.shape}") # debug 输出历史观测状态embedding的形状
            # print(f"encoder_in_tokens length before adding history_obs_state_embed: {len(encoder_in_tokens)}") # debug 输出添加历史观测状态embedding前encoder_in_tokens的长度:2
            history_obs_state_embed = history_obs_state_embed.permute(1, 0, 2)  # (N, B, D)
            encoder_in_tokens.extend(list(history_obs_state_embed))

            # print(f"encoder_in_tokens length after adding history_obs_state_embed: {len(encoder_in_tokens)}") # debug 输出添加历史观测状态embedding后encoder_in_tokens的长度:6




        # 新增：调用实例分割理解模块
        if self.config.use_segment_understanding:
            cam_key = f"observation.images.{self.config.seg_config.camera_name}"

            # 反归一化到 [0, 1] 范围，因为act的数据经过了mean-std归一化，不符合YOLO与FK输入需求
            norm_step:NormalizerProcessorStep = self.preprocessor.steps[3]
            imgs_for_yolo = norm_step._apply_transform(batch[cam_key], cam_key, FeatureType.VISUAL, inverse=True)
            obs_state_rad = norm_step._apply_transform(batch[OBS_STATE], OBS_STATE, FeatureType.STATE, inverse=True) * (torch.pi / 180.0) # 1、反归一化 2、度转弧度
            
            # 传入YOLO和FK，并得到分割理解的embedding
            yolo_r, yolo_mask = self.yolo_data_processer.get_yolo_data(imgs_for_yolo)
            ee_pose = self.kinematics.forward_kinematics_batch(obs_state_rad)
            segment_understanding_embed = self.segment_understanding_embedding(yolo_r, yolo_mask , ee_pose) #(B, D)
            # 最后把embedding放入tokens
            encoder_in_tokens.append(segment_understanding_embed)

        if self.config.use_history_token_replan_score:
            missing = [key for key in (OBS_STATE_HISTORY, ACTION_HISTORY, HISTORY_MASK) if key not in batch]
            if missing:
                raise KeyError(f"Missing history-token batch keys: {missing}")
            history_tokens, self.replan_aux_outputs = self.history_token_replan_score_model(
                state_history=batch[OBS_STATE_HISTORY],
                action_history=batch[ACTION_HISTORY],
                current_state=batch[OBS_STATE],
                history_mask=batch[HISTORY_MASK],
            )
            history_tokens = history_tokens.permute(1, 0, 2)  # [history_num_segments, B, D]
            encoder_in_tokens.extend(list(history_tokens))
            encoder_in_pos_embed.extend(list(self.history_token_replan_score_model.token_pos_embed.unsqueeze(1)))

        if self.config.use_key_history_token:
            missing = [key for key in (OBS_STATE_HISTORY, HISTORY_MASK) if key not in batch]
            if missing:
                raise KeyError(f"Missing key history batch keys: {missing}")
            key_history_tokens, self.key_history_aux_outputs = self.key_history_encoder(
                state_history=batch[OBS_STATE_HISTORY],
                current_state=batch[OBS_STATE],
                history_mask=batch[HISTORY_MASK],
            )
            key_history_tokens = key_history_tokens.permute(1, 0, 2)
            encoder_in_tokens.extend(list(key_history_tokens))
            encoder_in_pos_embed.extend(list(self.key_history_token_pos_embed.unsqueeze(1)))

        if self.config.image_features:
            # For a list of images, the H and W may vary but H*W is constant.
            # NOTE: If modifying this section, verify on MPS devices that
            # gradients remain stable (no explosions or NaNs).
            # print(list(batch.keys()))
            for img in batch[OBS_IMAGES]:
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)

                # Rearrange features to (sequence, batch, dim).
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                # Extend immediately instead of accumulating and concatenating
                # Convert to list to extend properly
                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

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
