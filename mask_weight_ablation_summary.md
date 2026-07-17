# Mask Weight v5 消融实验总结

更新时间：2026-07-17
数据目录：`E:\测试数据\消融实验`
最终模型：`mask_weight_v5`

## 1. 实验目的

消融实验用于验证 Mask Weight v5 中各核心组件是否真正提高实机夹取表现。所有消融均围绕同一条主线展开：利用 YOLO segmentation 引导 ACT 关注目标及其局部上下文，同时降低背景变化对动作预测的影响。

最终模型包含：

- confidence-weighted union soft mask；
- 空间 mask embedding；
- reliability gate；
- region attention；
- 2 个 target/context tokens；
- 2 层 object perceiver；
- 1 个 mask geometry token；
- 训练期 background counterfactual consistency。

## 2. 评估协议

- 测试目标放置条件：15 个。
- 每个条件最多执行 3 次夹取。
- 某条件一旦成功，停止该条件的后续尝试。
- `Success@1`：第一次机会内成功的条件数。
- `Success@2`：前两次机会内至少成功一次的条件数。
- `Success@3`：三次机会内至少成功一次的条件数。

该指标同时反映首次动作质量和允许有限重试后的任务完成能力。

## 3. 各消融配置

以下仅列出从 v5 完整版到对应消融版的改动。

### A1：去除 Background Consistency

```yaml
use_background_augmentation: false
use_background_consistency: false
background_consistency_loss_weight: 0.0
background_feature_consistency_loss_weight: 0.0
background_token_consistency_loss_weight: 0.0
```

验证训练期背景反事实约束是否能减少策略对无关背景的依赖。

### A2：去除 Target Tokens

```yaml
use_target_tokens: false
num_target_tokens: 0
```

验证显式目标区域摘要 token 是否为 ACT encoder 提供有效的物体中心表示。

### A3：去除 Mask Geometry Token

```yaml
use_mask_geometry_token: false
```

验证目标中心、尺度、面积与可靠性等显式几何条件是否有用。

### A4：去除 Region Attention

```yaml
use_region_attention: false
```

验证视觉位置对目标、邻域和背景区域原型的 cross-attention 是否有用。

### A5：去除 Target Token Attention

```yaml
use_target_token_attention: false
```

保留 masked pooled token，但移除 object perceiver refinement，用于区分“有 target token”和“对 target token 做视觉交互”的贡献。

## 4. 实机结果

| 模型 | 结构说明 | Success@1 | Success@2 | Success@3 |
|---|---|---:|---:|---:|
| Vanilla ACT | 不使用 Mask Weight | 3/15（20.0%） | 4/15（26.7%） | 6/15（40.0%） |
| A1 | 去除 Background Consistency | 4/15（26.7%） | 6/15（40.0%） | 7/15（46.7%） |
| A2 | 去除 Target Tokens | 4/15（26.7%） | 6/15（40.0%） | 8/15（53.3%） |
| A3 | 去除 Mask Geometry Token | 7/15（46.7%） | 9/15（60.0%） | 9/15（60.0%） |
| A4 | 去除 Region Attention | 5/15（33.3%） | 6/15（40.0%） | 8/15（53.3%） |
| A5 | 去除 Target Token Attention | 5/15（33.3%） | 7/15（46.7%） | 9/15（60.0%） |
| **Mask Weight v5** | **最终完整模型** | **12/15（80.0%）** | **13/15（86.7%）** | **14/15（93.3%）** |

## 5. 相对最终模型的下降

| 消融 | Success@1 下降 | Success@3 下降 | 解释 |
|---|---:|---:|---|
| A1 | 8/15 | 7/15 | 背景不变性约束是最关键组件 |
| A2 | 8/15 | 6/15 | 紧凑目标摘要显著帮助动作条件化 |
| A3 | 5/15 | 5/15 | 显式几何信息对定位和尺度判断有贡献 |
| A4 | 7/15 | 6/15 | 区域级交互优于仅靠局部空间 embedding |
| A5 | 7/15 | 5/15 | token 与视觉特征的进一步交互具有增益 |

这里的“下降”是条件数的描述性差值，不代表已经完成统计显著性检验。

## 6. 训练内部指标

后期日志摘要：

| 模型 | total loss 均值 | object delta ratio 均值 | object delta ratio P95 | reliability 均值 |
|---|---:|---:|---:|---:|
| A1 | 0.07020 | 0.356 | 0.486 | 0.0584 |
| A2 | 0.06989 | 0.000 | 0.000 | 0.0582 |
| A3 | 0.06998 | 0.394 | 0.597 | 0.0584 |
| A4 | 0.07003 | 0.391 | 0.580 | 0.0583 |
| A5 | 0.06996 | 0.000 | 0.000 | 0.0582 |
| **Mask Weight v5** | **0.06969** | **0.417** | **0.727** | **0.0581** |

解释：

- 各组 total loss 很接近，训练 loss 无法替代实机成功率。
- A2 和 A5 关闭对应分支，因此 object delta ratio 为 0，符合配置预期。
- v5 的后期均值 0.417 位于 0.3-0.5 的健康范围内。
- P95 反映偶发峰值，需结合动作稳定性判断，不能单独作为版本优劣标准。

## 7. 论文主表建议

建议正文使用下表：

| Method | BG Consistency | Target Tokens | Geometry Token | Region Attention | Token Attention | S@1 | S@2 | S@3 |
|---|:---:|:---:|:---:|:---:|:---:|---:|---:|---:|
| Vanilla ACT | - | - | - | - | - | 20.0 | 26.7 | 40.0 |
| w/o BG Consistency | - | ✓ | ✓ | ✓ | ✓ | 26.7 | 40.0 | 46.7 |
| w/o Target Tokens | ✓ | - | ✓ | ✓ | - | 26.7 | 40.0 | 53.3 |
| w/o Geometry Token | ✓ | ✓ | - | ✓ | ✓ | 46.7 | 60.0 | 60.0 |
| w/o Region Attention | ✓ | ✓ | ✓ | - | ✓ | 33.3 | 40.0 | 53.3 |
| w/o Token Attention | ✓ | ✓ | ✓ | ✓ | - | 33.3 | 46.7 | 60.0 |
| **Mask Weight v5** | **✓** | **✓** | **✓** | **✓** | **✓** | **80.0** | **86.7** | **93.3** |

表注建议：

> Success@k denotes the percentage of the 15 test conditions completed within the first k grasp attempts. Each condition allows at most three attempts and stops after the first success.

## 8. 可直接写入论文的结果描述

> 在 15 个统一目标放置条件下，Vanilla ACT 的 Success@1 和 Success@3 分别为 20.0% 和 40.0%，而 Mask Weight v5 分别达到 80.0% 和 93.3%。去除背景一致性、目标摘要 token、几何 token、区域注意力或目标 token 注意力后，Success@3 均下降至 46.7%-60.0%。其中，移除背景一致性造成最大下降，表明仅在结构中输入 mask 并不足以保证策略摆脱背景捷径；训练期背景反事实约束是提升泛化能力的关键。其他结构消融也均产生明显下降，说明空间、区域、目标摘要和几何信息在该任务中具有互补作用。

## 9. 结论边界

- 结果支持各组件对当前任务有正贡献。
- 15 个条件的规模仍较小，正文应报告原始计数和百分比。
- 不应仅根据训练 loss 推导实机性能。
- 不应将内部 delta ratio 描述为最终任务指标。
- 当前方法使用所有检测实例的 union mask，不具备任务条件化实例选择或显式类别识别能力。
