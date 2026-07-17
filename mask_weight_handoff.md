# customACT Mask Weight v5 Handoff

更新时间：2026-07-17
最终分支：`mask_weight_v5`
项目状态：方法与消融实验已收敛，v5 作为论文和后续实机测试的唯一最终版本。

## 1. 最终结论

Mask Weight v5 建立在稳定的 v4.2 核心结构上。方法使用 YOLO segmentation 生成的置信度加权 soft mask，在保留完整 RGB 主干的前提下，为 ACT 注入目标区域、邻域、几何与紧凑目标 token 信息，并通过训练期背景反事实一致性降低策略对无关背景的依赖。

最终版本不再沿用 v4.7 的 object-weighted mask 路线。该路线在训练后期将目标 token 分支增益放大到约 1.93，实机表现为动作摇摆；稳定结构的对应指标约为 0.3-0.5。后续不再通过叠加更强目标注入来追求收益。

论文与代码中统一称为：

> Mask Weight v5：YOLO 引导的多粒度视觉条件化 ACT。

## 2. 研究目标

原始 ACT 可以从固定采集环境中学习背景、光照和桌面纹理等捷径。Mask Weight 的目标不是删除背景，而是让策略更稳定地利用目标物体区域，同时在检测缺失或不可靠时仍保留原始 RGB 路径。

核心原则：

- YOLO 是软引导，不是硬门控。
- RGB 主干始终保留，不直接将背景置零。
- mask 信息通过多个低增益残差路径注入。
- 检测可靠性显式控制注入强度。
- 背景不变性只在训练阶段施加，推理时不增加第二次前向。
- 固定数量 token，避免检测实例数量直接改变 ACT 序列长度。

## 3. 最终结构

```text
RGB image
  |-- ResNet18 ------------------------------> visual feature map
  |
  `-- frozen YOLO segmentation -> soft mask ---------+
                                                      |
visual feature map + soft mask -> MaskGuidedVisualAdapter
  |-- guided visual features
  |-- 2 target/context tokens
  `-- 1 mask geometry token
                  |
                  v
ACT encoder -> ACT decoder -> action chunk
```

### 3.1 Soft mask

对第 `i` 个实例 mask `S_i` 和检测置信度 `c_i`：

```text
M_raw(x,y) = max_i(c_i * S_i(x,y))
M = clamp(GaussianBlur(M_raw), 0, 1)
```

默认 Gaussian kernel 为 7，sigma 为 2.0。没有 segmentation mask 但存在 box 时，以置信度填充 box 区域回退；完全没有检测时返回空 mask。

### 3.2 空间区域

将 soft mask 下采样到 ResNet 特征分辨率，构造：

- `T`：目标区域。
- `C`：目标 dilation 后减去目标本身得到的邻域。
- `B`：目标与邻域之外的背景。
- `Q`：由最大 mask 置信度扩展得到的 confidence map。

四通道 `[T,C,B,Q]` 经卷积编码，以残差方式加到视觉特征上。

### 3.3 Reliability gate

可靠性由 mask 最大置信度与面积共同决定。空 mask 的可靠性为 0；非空 mask 使用 0.6 的 floor，再根据置信度和面积质量上调。空间注入、区域注意力、目标 token refinement 和几何 token 都受该值控制。

### 3.4 Region attention

分别对目标、邻域和背景区域做 masked pooling，形成 3 个区域原型。视觉 token 对区域原型做 cross-attention，输出只在目标与邻域位置以可学习小门控残差注入。默认背景空间权重为 0。

### 3.5 Target tokens

最终固定使用 2 个 token：

- 目标区域 pooled token。
- 邻域 pooled token。

token 经过小型投影后，由 2 层 object perceiver 对视觉 token 做带 mask bias 的 cross-attention。增量乘以可靠性和初值为 0.2 的可学习 gate，再送入 ACT encoder。

### 3.6 Geometry token

1 个 geometry token 显式编码 8 维 mask 几何量：

```text
[center_x, center_y, spread_x, spread_y,
 area, confidence, reliability, context_area]
```

几何向量经 MLP 投影到 ACT 隐空间，并由可靠性与可学习 gate 缩放。

### 3.7 Background consistency

训练时，以 0.5 概率保持目标及其较大邻域不变，只替换背景。原图和增强图共享同一个 ACT latent sample，约束：

- 动作 chunk 一致。
- 目标区域视觉表示一致。
- 目标 token 一致。

默认权重分别为 0.1、0.03、0.03。推理阶段只有真实 RGB 的一次前向。

## 4. 最终默认配置

```yaml
use_spatial_embedding: true
use_residual_gate: false
use_region_attention: true
use_reliability_gate: true
use_target_tokens: true
num_target_tokens: 2
use_target_token_attention: true
target_object_perceiver_layers: 2
use_mask_geometry_token: true

gate_init: 0.2
region_attention_gate_init: 0.2
target_token_attention_gate_init: 0.2
mask_geometry_token_gate_init: 0.2
reliability_floor: 0.6

use_background_augmentation: true
use_background_consistency: true
background_aug_p: 0.5
background_consistency_loss_weight: 0.1
background_feature_consistency_loss_weight: 0.03
background_token_consistency_loss_weight: 0.03
```

## 5. 消融实验结论

每个模型在相同 15 个目标放置条件下测试；每个条件最多执行三次夹取，成功后停止后续尝试。

| 模型 | 改动 | Success@1 | Success@2 | Success@3 |
|---|---|---:|---:|---:|
| Vanilla ACT | 不使用 Mask Weight | 3/15 | 4/15 | 6/15 |
| A1 | 去除 Background Consistency | 4/15 | 6/15 | 7/15 |
| A2 | 去除 Target Tokens | 4/15 | 6/15 | 8/15 |
| A3 | 去除 Geometry Token | 7/15 | 9/15 | 9/15 |
| A4 | 去除 Region Attention | 5/15 | 6/15 | 8/15 |
| A5 | 去除 Target Token Attention | 5/15 | 7/15 | 9/15 |
| **Mask Weight v5** | **最终完整模型** | **12/15** | **13/15** | **14/15** |

主要结论：

- v5 的 Success@3 为 93.3%，Vanilla ACT 为 40.0%。
- v5 的 Success@1 为 80.0%，说明提升不仅来自更多重试机会。
- Background Consistency 的消融下降最大，是抑制背景捷径的关键训练机制。
- Target Tokens、Region Attention、Geometry Token 和 Target Token Attention 均有正贡献。
- 现有样本量适合报告描述性结果，不应过度宣称统计显著性。

## 6. 健康指标

重点监控：

- `adapter_object_perceiver_delta_ratio`：建议后期约 0.3-0.5。
- `adapter_output_delta_ratio`：监控 mask adapter 总注入是否持续放大。
- `adapter_reliability_mean`：判断 YOLO 引导是否频繁失效。
- `mask_weight/consistency/action_delta_l1`：判断背景变化对动作输出的影响。
- mask coverage、confidence、target/background RMS gain：辅助排查检测质量和能量分配。

指标用于诊断，不作为自动追逐的优化目标。最终判断仍以同协议实机任务成功率和动作稳定性为准。

## 7. 代码位置

- `src/lerobot/policies/customACT/modeling_customACT.py`：YOLO、adapter、ACT token 序列和训练损失集成。
- `src/lerobot/policies/customACT/mask_weight/mask_weight.py`：soft mask、背景增强、视觉 adapter、target tokens 和 geometry token。
- `src/lerobot/policies/customACT/mask_weight/configuration_mask_weight.py`：最终默认配置与参数校验。
- `tests/policies/test_customact_mask_weight.py`：配置、形状、梯度、空 mask、背景增强和调试指标测试。

训练环境：

```powershell
conda activate lerobot
python -m pytest tests/policies/test_customact_mask_weight.py -q
```

## 8. 后续边界

- 不再继续 v4.7 的 object weighting 路线。
- 不引入可变数量 instance tokens 或类别 token，除非有新的独立实验证据。
- 不把训练 loss 的细微变化等同于实机控制提升。
- 新改动必须同时检查注入比例、动作平滑性与 Success@k。
- 论文和代码统一使用 `mask_weight_v5`，不再使用旧候选命名指代最终模型。
