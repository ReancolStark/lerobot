# customACT Mask Weight v5 项目总结与论文写作材料

更新时间：2026-07-17
最终代码分支：`mask_weight_v5`
训练环境：`conda activate lerobot`

---

## 1. 文档定位

本文系统总结 LeRobot `customACT` 中 Mask Weight v5 的研究动机、方法结构、代码实现、训练目标、推理流程、稳定性设计、消融结果、论文创新点、局限性与复现实验要求。后续论文、答辩材料和实机测试均以 v5 为唯一最终模型名称。

本项目解决的问题是：如何让 ACT 利用 YOLO segmentation 提供的目标区域先验，更关注与操作相关的物体及其局部环境，并降低固定背景、桌面纹理、光照和无关物体对动作预测的影响。

## 2. 一句话概括

Mask Weight v5 是一种 YOLO 引导的多粒度 ACT 视觉条件化方法：它保留完整 RGB 主干，将 confidence-weighted soft mask 转换为稠密空间引导、区域注意力、紧凑目标 token 和显式几何 token，并在训练阶段通过背景反事实一致性约束，使策略在目标保持不变而背景变化时维持稳定动作与目标表示。

可用于英文摘要的方法概括：

> We introduce a YOLO-guided multi-granularity visual conditioning module for ACT. The method preserves the full RGB pathway and injects soft spatial, regional, object-centric, and geometric cues into visual features. A background counterfactual consistency objective further encourages stable actions and target representations under background changes.

## 3. 研究问题

### 3.1 原始 ACT 的背景捷径

ACT 从 RGB 图像和机器人状态直接预测一段动作序列。在采集背景稳定、相机固定、桌面纹理单一的数据集中，模型可能将背景相关模式作为动作预测捷径，而没有充分学习目标物体本身。

这种捷径会导致：

- 背景、光照或无关物体变化时成功率下降；
- 训练 loss 正常，但实机动作不稳定；
- 即使额外提供 segmentation，模型仍可能忽略它并继续依赖 RGB 背景；
- 过强的检测信息注入又可能把 YOLO 的抖动放大为控制抖动。

### 3.2 为什么不直接硬掩码

最直接的方案是：

```text
masked_image = image * mask
```

或：

```text
masked_feature = feature * mask
```

v5 没有采用这种方案，原因包括：

- 直接置黑会制造新的输入分布；
- ResNet 已经混合局部上下文，后置硬掩码无法逆转早期卷积中的信息融合；
- 漏检或边缘截断会直接删除有效视觉信息；
- 操作不仅依赖物体像素，还依赖夹爪、接触边界和目标附近上下文；
- 硬门控会让策略对检测波动过度敏感。

因此，v5 的基本原则是“YOLO 作为软引导，RGB 作为稳定主干”。

## 4. 方法创新点

### 4.1 保留 RGB 主干的多粒度条件化

模型不改变原始 RGB 输入，不删除背景，而是在 ResNet 特征之后并行引入四类信息：

1. 稠密空间 embedding：描述每个特征位置属于目标、邻域还是背景。
2. Region attention：让视觉位置从目标、邻域和背景区域原型中提取上下文。
3. Target tokens：将目标和邻域压缩为固定数量的 ACT encoder token。
4. Geometry token：显式编码目标中心、尺度、面积、置信度和可靠性。

这种设计在保留场景细节的同时，让目标先验能以不同粒度参与动作预测。

### 4.2 背景反事实一致性训练

训练时保持目标及其邻域不变，只替换背景，并要求原图与增强图产生一致的：

- 动作 chunk；
- 目标区域视觉表示；
- 目标摘要 token。

它将“降低背景依赖”从结构期待转化为直接可优化目标。消融中移除此机制后，Success@3 从 14/15 降至 7/15，是其有效性的直接实验证据。

### 4.3 检测可靠性控制的软回退

所有主要 mask 分支均由 reliability 调节。完全没有检测时，mask-dependent 增量关闭，模型退回接近原始 ACT 的视觉路径；检测存在时，再根据置信度和面积质量决定注入强度。

### 4.4 固定长度的目标 token 表示

模型将所有检测实例先合并为 union mask，再压缩为固定的 2 个 target/context tokens 和 1 个 geometry token。ACT encoder 的序列长度不随检测数量变化，从而避免实例数量、排序和漏检直接造成 token 序列波动。

### 4.5 小门控残差与稳定初始化

多个投影层采用零初始化或小方差初始化，关键残差 gate 初值为 0.2。训练开始时模型接近原始视觉路径，再逐步学习使用 mask 信息，降低额外分支在初始化阶段破坏特征分布的风险。

## 5. 总体结构

```text
Normalized RGB
    |
    +-----------------------> ResNet18 --------------------+
    |                                                     |
    +-> inverse normalize -> Frozen YOLO Segmentation     |
                                  |                       |
                                  v                       v
                    confidence-weighted soft mask   visual features
                                  |                       |
                                  +-----------+-----------+
                                              v
                               MaskGuidedVisualAdapter
                                              |
               +------------------------------+-------------------------+
               |                              |                         |
               v                              v                         v
       guided visual features           2 target tokens          geometry token
          [B,D,Hf,Wf]                     [2,B,D]                   [1,B,D]
               |                              |                         |
               +------------------------------+-------------------------+
                                              v
           [latent/state tokens, visual tokens, target tokens, geometry token]
                                              |
                                         ACT encoder
                                              |
                                         ACT decoder
                                              |
                                      action chunk prediction
```

## 6. 基础 ACT

Mask Weight v5 不替换 ACT 主体，而是在 ACT 视觉编码前增加条件化模块。基础流程仍包括：

- ResNet18 提取每个相机的视觉特征；
- VAE encoder 在训练时根据机器人状态和真实动作产生 latent；
- Transformer encoder 融合 latent、状态和视觉 token；
- Transformer decoder 使用 learned queries 一次预测 action chunk；
- 推理时 latent 置零，不需要真实动作序列。

基础训练项可写为：

```text
L_ACT = L_action + beta_KL * L_KL
```

其中 `L_action` 是忽略 padding 后的动作 L1 重建项，默认 `beta_KL = 10.0`。

## 7. YOLO Soft Mask

### 7.1 输入与冻结方式

模型从 LeRobot normalizer 的输出中反归一化 RGB 到 `[0,1]`，送入冻结的 YOLO segmentation 模型。YOLO 不参与 ACT 反向传播，其作用是提供中间视觉先验。

### 7.2 多实例合并

对第 `i` 个实例分割结果 `S_i` 和置信度 `c_i`：

```text
M_raw(x,y) = max_i [c_i * S_i(x,y)]
```

然后进行 Gaussian blur：

```text
M(x,y) = clamp(GaussianBlur(M_raw), 0, 1)
```

默认 kernel size 为 7，sigma 为 2.0。逐像素最大值保留最强实例证据，blur 缓和分割边缘的离散变化。

### 7.3 回退行为

- 有 segmentation mask：优先使用 mask。
- 无 mask 但有 bounding box：以检测置信度填充 box 区域。
- 无任何检测：返回全零 mask。

### 7.4 表达边界

v5 使用 union mask，不将 YOLO 类别、实例身份或可变数量实例 token 输入 ACT。因此它学习的是“被检测目标区域的通用操作先验”，不是显式物品类别识别，也不是任务条件化的实例选择。

## 8. 空间引导与 Reliability

### 8.1 三个区域

将 soft mask 下采样到 ResNet feature map 尺寸，记为 `T`。用 dilation radius 3 构造：

```text
D = MaxPool(T, kernel_size=7)
C = clamp(D - T, 0, 1)
B = clamp(1 - D, 0, 1)
```

其中：

- `T` 是 target region；
- `C` 是 target 周围的 context ring；
- `B` 是远离 target 和 context 的 background region。

再将最大 mask 值扩展为空间常量图 `Q`：

```text
Q(x,y) = max_xy T
G = concat(T, C, B, Q)
```

### 8.2 Reliability 计算

令：

```text
q = max(T)
a = mean(T)
a_min = 0.002
a_max = 0.65
```

面积质量为：

```text
s_low  = clamp(a / a_min, 0, 1)
s_high = clamp((1-a) / (1-a_max), 0, 1)
s_area = s_low * s_high
r_raw  = clamp(q * s_area, 0, 1)
```

最终：

```text
r = 0,                    if a = 0
r = 0.6 + 0.4 * r_raw,    if a > 0
```

空 mask 明确关闭引导；非空 mask 保持至少 0.6 的引导强度，同时根据检测质量上调。

### 8.3 稠密空间 embedding

四通道 guidance 经 `3x3 Conv -> GELU -> 1x1 Conv` 投影到视觉维度：

```text
Delta_spatial = r * E(G)
F_spatial = F_rgb + Delta_spatial
```

最后一层卷积零初始化，所以训练起点不会突然改变 ResNet 输出。

## 9. Region Attention

### 9.1 区域原型

对特征 `F_spatial` 分别按 `T`、`C`、`B` 做 masked average pooling：

```text
p_t = Pool(F_spatial, T)
p_c = Pool(F_spatial, C)
p_b = Pool(F_spatial, B)
```

三个 pooled vectors 经投影并加入 region type embedding，成为区域原型 token。

### 9.2 Cross-attention

展平的视觉 token 作为 query，区域原型作为 key/value：

```text
A_region = MHA(Q=visual_tokens, K=region_tokens, V=region_tokens)
```

空间权重为：

```text
W_region = clamp(T + 0.3*C + 0.0*B, 0, 1)
```

最终：

```text
Delta_region = g_region * r * W_region * Proj(A_region)
F_guided = F_spatial + Delta_region
```

其中 `g_region` 的初值为 0.2。背景原型可以作为注意力上下文，但背景位置默认不接收 region residual。

## 10. Target Tokens 与 Object Perceiver

### 10.1 初始 token

最终固定使用两个 pooled token：

```text
z_target  = Pool(F_guided, T)
z_context = Pool(F_guided, C)
```

二者经 `LayerNorm -> Linear -> GELU -> Linear` 的残差投影，并加入独立可学习位置 embedding。

### 10.2 两层 object perceiver

每层包括：

1. 两个 object tokens 之间的 self-attention；
2. object tokens 对全部视觉 token 的 cross-attention；
3. 维度为 1024 的 FFN；
4. 每个子层的 residual connection 与 normalization。

cross-attention bias 来自对应 token 的空间 mask，默认 bias scale 为 2.0。目标 token 更偏向目标区域，邻域 token 更偏向 context ring，但仍能读取全局视觉特征。

refinement 增量由 gate 和 reliability 控制：

```text
Delta_obj = z_refined - z_seed
z_out = z_seed + g_obj * r * Delta_obj
```

`g_obj` 初值为 0.2。后期 `object_perceiver_delta_ratio` 的健康均值约为 0.3-0.5；v5 实验均值为 0.417。

## 11. Mask Geometry Token

从 target mask 计算 8 维几何向量：

```text
g = [center_x,
     center_y,
     spread_x,
     spread_y,
     target_area,
     confidence,
     reliability,
     context_area]
```

其中中心和扩展范围均在 `[0,1]` 坐标系计算。空 mask 时中心回退到 `(0.5,0.5)`，扩展与面积为 0。

几何向量通过：

```text
LayerNorm(8) -> Linear(8,D) -> GELU -> Linear(D,D)
```

映射为 1 个 ACT token：

```text
z_geometry = g_geometry * r * MLP(g)
```

最后一层零初始化，gate 初值为 0.2。该 token 直接给 ACT encoder 提供目标位置和尺度信息，避免所有几何关系都必须从卷积特征中隐式恢复。

## 12. ACT Token 序列集成

对每个相机，MaskGuidedVisualAdapter 返回：

- `guided_features: [B,D,Hf,Wf]`；
- `target_tokens: [2,B,D]`；
- `geometry_token: [1,B,D]`；
- 对应的位置 embedding。

视觉特征展平为 `Hf*Wf` 个 token 后，立即追加该相机的 2 个 target tokens 和 1 个 geometry token。最终 encoder 序列包含：

```text
[latent token,
 robot-state token,
 camera-1 visual tokens,
 camera-1 target tokens,
 camera-1 geometry token,
 ...]
```

这种做法使目标条件与对应相机视觉 token 保持局部组织关系，同时维持固定序列长度。

## 13. Background Counterfactual Consistency

### 13.1 背景构造

训练时用较大的 dilation 保护目标与邻域：

```text
K = Dilate(M, radius=21)
```

低于 0.05 的区域不计入保护范围。背景替换候选包括随机颜色或像素噪声，mixed 模式下二者各有 0.5 概率。增强强度在 0.75 到 1.0 之间采样，并以 0.5 概率应用到样本。

```text
I_aug = K * I + (1-K) * Mix(I, I_background, strength)
```

没有有效目标 mask 的样本保持原样，避免在检测失败时无依据地破坏整张图像。

### 13.2 双前向流程

训练 batch 先执行真实图像前向，再对增强图像执行第二次前向。两次前向共享同一个 VAE latent sample，避免随机 latent 差异被误计为背景敏感性。

增强前向强制复用第一次前向生成的 YOLO mask，不再次运行 YOLO，从而确保比较只改变背景。

### 13.3 三层一致性目标

动作一致性：

```text
L_action_cons = mean_valid(|A_aug - stopgrad(A_real)|)
```

目标特征一致性：

```text
L_feature_cons = |P_target(F_aug) - stopgrad(P_target(F_real))|_1
```

目标 token 一致性：

```text
L_token_cons = |Z_aug - stopgrad(Z_real)|_1
```

总训练目标：

```text
L_total = L_action
        + 10.0 * L_KL
        + 0.10 * L_action_cons
        + 0.03 * L_feature_cons
        + 0.03 * L_token_cons
```

## 14. 训练与推理差异

| 项目 | 训练 | 推理 |
|---|---|---|
| 真实 RGB 前向 | 是 | 是 |
| YOLO segmentation | 是 | 是 |
| MaskGuidedVisualAdapter | 是 | 是 |
| 背景替换 | 是 | 否 |
| 第二次 ACT 前向 | 是 | 否 |
| 一致性训练项 | 是 | 否 |
| VAE latent 来源 | 动作条件化采样 | 零向量 |

因此，背景增强仅增加训练成本，不增加实机推理时的第二次 ACT 前向。

## 15. 最终关键配置

```yaml
use_spatial_embedding: true
use_residual_gate: false
use_region_attention: true
use_reliability_gate: true

adapter_hidden_dim: 128
mask_dropout_p: 0.1
mask_blur_kernel_size: 7
mask_blur_sigma: 2.0

reliability_min_area: 0.002
reliability_max_area: 0.65
reliability_floor: 0.6

region_attention_heads: 4
region_attention_gate_init: 0.2
region_attention_context_weight: 0.3
region_attention_background_weight: 0.0

use_target_tokens: true
num_target_tokens: 2
use_target_token_attention: true
target_token_attention_heads: 4
target_token_attention_gate_init: 0.2
target_token_mask_bias_scale: 2.0
target_object_perceiver_layers: 2
target_object_perceiver_ffn_dim: 1024
target_object_perceiver_dropout: 0.0

use_mask_geometry_token: true
mask_geometry_token_gate_init: 0.2

use_background_augmentation: true
use_background_consistency: true
background_aug_p: 0.5
background_consistency_loss_weight: 0.1
background_feature_consistency_loss_weight: 0.03
background_token_consistency_loss_weight: 0.03
background_aug_context_dilation: 21
background_aug_keep_threshold: 0.05
background_aug_mode: mixed
background_consistency_aug_min_strength: 0.75
background_consistency_aug_max_strength: 1.0
```

最终配置中 `mask_noise_p = 0.0`、`mask_noise_dropout_p = 0.0`，因此训练默认不主动扰动 mask；`mask_dropout_p = 0.1` 仍用于引导分支的整体 dropout 回退。

## 16. 参数开销

在 `dim_model = 512` 的当前实现中，MaskGuidedVisualAdapter 约包含 9,016,979 个可训练参数，主要分布为：

| 模块 | 参数量 |
|---|---:|
| Spatial embedding | 70,784 |
| Region attention | 1,842,177 |
| Target token 与 object perceiver | 6,836,225 |
| Geometry token | 267,793 |
| **合计** | **9,016,979** |

object perceiver 是参数开销最大的部分，因此后续若做模型压缩，应优先评估层数、FFN 维度和共享方式，而不是删除低成本 geometry token。

## 17. 调试与稳定性指标

代码记录以下关键指标：

| 指标 | 含义 |
|---|---|
| `adapter_visual_rms` | 原始视觉特征 RMS |
| `adapter_spatial_delta_ratio` | 空间 embedding 增量相对视觉特征的比例 |
| `adapter_region_attention_delta_ratio` | 区域注意力增量比例 |
| `adapter_object_perceiver_delta_ratio` | 目标 token refinement 增量比例 |
| `adapter_output_delta_ratio` | adapter 总输出增量比例 |
| `adapter_reliability_mean` | 当前 batch 的平均引导可靠性 |
| `adapter_target_background_rms_ratio_gain` | 引导前后目标/背景能量比增益 |
| `mask_weight/consistency/action_delta_l1` | 背景变化导致的动作差异 |

健康区间不是硬阈值，但已有实验表明：

- `object_perceiver_delta_ratio` 后期均值约 0.3-0.5 时较稳定；
- 指标接近 2 时，目标 token 分支可能成为高增益噪声源；
- 训练 loss 正常、没有 NaN，并不能排除实机摇摆；
- 内部指标必须与动作平滑性和 Success@k 联合判断。

## 18. 消融实验

### 18.1 协议

所有模型在相同的 15 个目标放置条件下测试。每个条件最多执行三次夹取，成功后停止后续尝试。

```text
Success@1 = 第一次机会内成功的条件数 / 15
Success@2 = 前两次机会内至少成功一次的条件数 / 15
Success@3 = 三次机会内至少成功一次的条件数 / 15
```

### 18.2 结果表

| 模型 | 改动 | Success@1 | Success@2 | Success@3 |
|---|---|---:|---:|---:|
| Vanilla ACT | 不使用 Mask Weight | 3/15（20.0%） | 4/15（26.7%） | 6/15（40.0%） |
| A1 | 去除 Background Consistency | 4/15（26.7%） | 6/15（40.0%） | 7/15（46.7%） |
| A2 | 去除 Target Tokens | 4/15（26.7%） | 6/15（40.0%） | 8/15（53.3%） |
| A3 | 去除 Geometry Token | 7/15（46.7%） | 9/15（60.0%） | 9/15（60.0%） |
| A4 | 去除 Region Attention | 5/15（33.3%） | 6/15（40.0%） | 8/15（53.3%） |
| A5 | 去除 Target Token Attention | 5/15（33.3%） | 7/15（46.7%） | 9/15（60.0%） |
| **Mask Weight v5** | **最终完整模型** | **12/15（80.0%）** | **13/15（86.7%）** | **14/15（93.3%）** |

### 18.3 结果解释

1. v5 将 Success@3 从 Vanilla ACT 的 40.0% 提升到 93.3%。
2. v5 的 Success@1 达到 80.0%，说明收益不只是依赖重试。
3. A1 的下降最大，说明显式训练背景不变性比单纯把 mask 输入网络更关键。
4. A2 和 A4 均降至 53.3%，说明紧凑目标 token 与区域级交互是互补的。
5. A3 和 A5 均降至 60.0%，支持几何 token 与 token refinement 的独立贡献。
6. 所有结构消融均低于 v5，形成较完整的组件有效性证据链。

### 18.4 训练日志摘要

| 模型 | total loss 均值 | object delta ratio 均值 | object delta ratio P95 | reliability 均值 |
|---|---:|---:|---:|---:|
| A1 | 0.07020 | 0.356 | 0.486 | 0.0584 |
| A2 | 0.06989 | 0.000 | 0.000 | 0.0582 |
| A3 | 0.06998 | 0.394 | 0.597 | 0.0584 |
| A4 | 0.07003 | 0.391 | 0.580 | 0.0583 |
| A5 | 0.06996 | 0.000 | 0.000 | 0.0582 |
| **v5** | **0.06969** | **0.417** | **0.727** | **0.0581** |

训练 loss 几乎无法区分这些模型，而实机成功率差异显著。因此论文应以实机任务结果为主，日志指标用于解释和诊断。

## 19. 版本路线判断

项目曾尝试更强的 object weighting 和更细的实例信息注入。该路线在离线日志中看似增强了目标分支，但实机动作出现摇摆。关键证据是 object perceiver 增量从健康水平被放大到约 1.93。

由此得到的工程结论：

- 目标信息不是越强越好；
- segmentation 噪声经过高增益分支会转化为控制噪声；
- 单纯为压低某个指标继续增加 gate、loss 或 normalization，容易形成反复补丁；
- 最终版本应优先保留已经通过实机验证的结构整合与注入强度。

v5 因此选择稳定、完整、可解释的多粒度结构，不再继续 object-weighted mask 扩张路线。

## 20. 代码实现映射

### 20.1 `modeling_customACT.py`

负责：

- 调用冻结 YOLO；
- 将 mask 与 ResNet feature map 对齐；
- 调用 `MaskGuidedVisualAdapter`；
- 将 target tokens 和 geometry token 插入 ACT encoder 序列；
- 构造训练期背景增强 batch；
- 复用 latent 与 forced mask；
- 计算动作、目标特征和目标 token 的背景一致性；
- 收集 debug metrics。

### 20.2 `mask_weight.py`

负责：

- `yolo_result_to_soft_mask`；
- `make_mask_guided_background_augmentation`；
- `MaskGuidedObjectPerceiverLayer`；
- `MaskGuidedVisualAdapter`；
- guidance、reliability、masked pooling、region attention、target tokens、geometry token；
- adapter 内部稳定性指标。

### 20.3 `configuration_mask_weight.py`

集中定义最终开关、gate、mask 后处理、可靠性、背景增强和日志配置，并在 `__post_init__` 中检查范围、head 数和强度区间。

### 20.4 `test_customact_mask_weight.py`

覆盖：

- v5 默认配置；
- adapter 输出形状与梯度；
- target tokens 和 geometry token；
- reliability 和空 mask 回退；
- region attention 与 object perceiver 调试指标；
- 背景增强保留目标并改变背景；
- 配置非法值校验。

## 21. 可复现流程

### 21.1 环境

```powershell
conda activate lerobot
```

### 21.2 单元测试

```powershell
python -m pytest tests/policies/test_customact_mask_weight.py -q
```

### 21.3 训练记录要求

每次实验至少保存：

- 完整 config snapshot；
- Git branch 和 commit id；
- `mask_weight_debug.jsonl`；
- YOLO 权重路径与置信度阈值；
- 数据集版本和相机列表；
- 随机种子、训练步数和 checkpoint；
- 实机 Success@1、Success@2、Success@3 原始计数；
- 失败类型和动作稳定性观察。

### 21.4 公平比较要求

消融组除目标组件外应保持：

- 相同数据集；
- 相同训练步数与优化器；
- 相同 YOLO 权重；
- 相同随机种子策略；
- 相同实机目标放置条件；
- 相同最大尝试次数与停止规则。

## 22. 论文方法章节建议结构

### 22.1 Problem Formulation

介绍视觉模仿学习中的背景捷径、YOLO segmentation 先验和 ACT action chunking。

### 22.2 Confidence-Weighted Mask Construction

给出实例置信度加权、union 合并、Gaussian softening 与空检测回退。

### 22.3 Reliability-Gated Multi-Granularity Conditioning

依次介绍 spatial embedding、region attention、target tokens 和 geometry token，并说明所有分支受 reliability 控制。

### 22.4 Background Counterfactual Consistency

说明背景替换方法、共享 latent、forced mask 和三层一致性训练目标。

### 22.5 Integration with ACT

说明 token 序列、训练与推理差异、计算成本与固定序列长度。

## 23. 论文贡献表述建议

建议写成三点：

1. 提出一种保留完整 RGB 主干的 YOLO 引导多粒度视觉条件化模块，将稠密空间、区域、目标摘要和几何信息统一注入 ACT。
2. 设计可靠性门控与固定数量目标 token，使模型在检测缺失时软回退，并减少实例数量波动对控制序列的干扰。
3. 引入训练期背景反事实一致性，在动作、目标视觉表示和目标 token 三个层次降低背景依赖，并通过实机消融验证其贡献。

避免使用以下过强表述：

- “模型理解了物品类别。”
- “模型能够从多个实例中选择任务相关实例。”
- “方法完全消除了背景影响。”
- “所有提升都具有统计显著性。”
- “训练 loss 更低证明实机更好。”

## 24. 可直接用于论文的中文方法描述

> 为降低 ACT 对训练场景背景的依赖，本文提出 Mask Weight v5，一种由 YOLO 实例分割引导的多粒度视觉条件化方法。该方法不对 RGB 图像进行硬掩码，而是在卷积视觉特征上并行注入稠密空间引导、区域级注意力、目标摘要 token 与几何 token。首先，YOLO 实例分割结果按检测置信度加权并合并为 soft union mask；随后，模型从中构造目标、邻域和背景区域，并根据 mask 置信度与面积估计可靠性。所有附加分支均采用小门控残差并受可靠性调节，从而在检测缺失时退回接近原始 ACT 的视觉路径。训练阶段进一步保持目标及其邻域不变、随机替换背景，并约束原图与增强图的动作序列、目标区域特征和目标 token 保持一致。该设计使 segmentation 既能提供目标先验，又不会成为高增益硬门控输入。

## 25. 可直接用于论文的实验描述

> 在 15 个统一目标放置条件下，每个条件最多允许三次夹取并在首次成功后停止。Vanilla ACT 的 Success@1 和 Success@3 分别为 20.0% 和 40.0%，Mask Weight v5 分别达到 80.0% 和 93.3%。去除背景一致性、目标 token、几何 token、区域注意力或目标 token 注意力后，Success@3 分别下降至 46.7%、53.3%、60.0%、53.3% 和 60.0%。其中，去除背景一致性导致最大性能下降，说明显式约束策略在目标不变、背景变化时保持动作稳定，是减少背景捷径的关键。其余组件的消融也均造成性能下降，表明稠密空间、区域上下文、目标摘要与几何信息具有互补作用。

## 26. 局限性

### 26.1 依赖外部检测器

YOLO 漏检、误检、边缘偏移和置信度变化仍会影响策略。reliability gate 可以缓和影响，但不能恢复缺失的目标信息。

### 26.2 Union mask 不区分类别和实例

多个检测实例被合并后，模型不知道哪个实例属于哪一类别，也不能根据任务语言或指令选择实例。当前结果只支持目标区域引导，不支持语义理解主张。

### 26.3 训练计算成本增加

背景一致性需要第二次 ACT 前向，训练时间和显存开销增加。推理阶段没有该额外前向。

### 26.4 实验规模有限

15 个测试条件能够形成清楚的描述性差异，但仍不足以支撑强统计结论。论文应同时报告成功计数和百分比，并清楚说明停止规则。

### 26.5 参数主要集中在 object perceiver

adapter 约 9.02M 参数，其中约 6.84M 来自 target token 分支。若部署资源受限，需要独立评估压缩后的成功率与稳定性。

## 27. 后续研究方向

合理的后续方向包括：

- 在不改变最终论文主线的前提下，用更多场景和背景验证泛化；
- 增加不同 YOLO 质量、漏检率和边界噪声下的鲁棒性评估；
- 对 object perceiver 做参数共享或轻量化；
- 使用多随机种子与更多实机条件估计置信区间；
- 将动作抖动、轨迹平滑度和成功率纳入统一评估。

不建议在当前最终模型上继续叠加强目标权重、多实例 token 或类别 token，除非新实验明确证明收益并保持注入比例稳定。

## 28. 最终定义

Mask Weight v5 的最终结构为：

```text
Frozen YOLO segmentation
+ confidence-weighted union soft mask
+ reliability-gated spatial embedding
+ target/context/background region attention
+ 2 target/context tokens
+ 2-layer object perceiver
+ 1 mask geometry token
+ action/feature/token background consistency during training
+ original ACT action chunking backbone
```

项目最终判断是：Mask Weight 的有效性来自多个低增益组件的互补整合，以及训练期对背景不变性的直接约束，而不是不断增强单一目标分支。v5 在当前实机协议下达到 14/15 的 Success@3，并且各核心组件均得到消融支持，因此可作为论文、代码发布和后续复现实验的最终版本。
