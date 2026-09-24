# 载荷转移轮地力验证模型说明

本文档说明由独立验证脚本辨识出来的低阶轮地力模型：

```bash
tools/validate_hf_terramechanics_flat.py
```

该模型只用于诊断和验证，用来比较“计算得到的轮地力”和 HF 数据中的轮地力。  
它不会修改 V3 模型训练、推理、checkpoint 选择逻辑，也不会修改数据预处理流程。

## 模型目的

当前验证结果表明，HF 轮地力不能很好地由传统 Bekker / sinkage 土力模型解释。  
目前效果最好的低阶模型是“载荷转移模型”：

```text
Fz = 每轮静态支撑载荷 + 车体姿态/加速度导致的载荷转移
Fx = 每轮切向力均值
Fy = 每轮侧向力均值
```

因此，该模型应理解为 rover 六轮支撑载荷分配模型，而不是土壤压强积分模型。

## 模型形式

对每个轮子 `i`，当前选用模型为：

```text
Fx_i = b_Fx_i

Fy_i = b_Fy_i

Fz_i = b_Fz_i
     + k_roll_i  * roll
     + k_pitch_i * pitch
     + k_ax_i    * acc_x
     + k_ay_i    * acc_y
     + k_az_i    * acc_z
     + k_vz_i    * relu(-wheel_vz_i)
```

其中：

```text
i = 0..5
```

也就是六个轮子分别有自己的参数。

## 当前消融结果

当前消融实验显示，`load_transfer_only` 优于每轮均值模型，也优于加入 Bekker 的模型：

```text
wheel_mean:
  vector_rmse = 100.34
  Fz_rmse     = 93.68
  Fz_corr     = 0.340

load_transfer_only:
  vector_rmse = 84.63
  Fz_rmse     = 76.64
  Fz_corr     = 0.642

load_transfer + Bekker:
  vector_rmse = 85.76
  Fz_rmse     = 77.68
  Fz_corr     = 0.637
```

因此，Bekker 不属于当前选用的低阶力模型主路径。

## 输入量

该模型使用以下观测状态量：

```text
roll
pitch
acc_x
acc_y
acc_z
wheel_vz_i
wheel_id
```

在验证脚本中，这些列从 merged CSV 中读取：

```text
roll  -> 优先 hf_roll，  不存在时使用 lf_roll
pitch -> 优先 hf_pitch， 不存在时使用 lf_pitch
acc_x -> 优先 hf_acc_x， 不存在时使用 lf_acc_x
acc_y -> 优先 hf_acc_y， 不存在时使用 lf_acc_y
acc_z -> 优先 hf_acc_z， 不存在时使用 lf_acc_z
```

`wheel_vz_i` 来自 `--velocity_source` 选择的轮子速度。  
在 `--sph_aligned` 模式下，轮子速度会先旋转到轮子局部坐标系。

## 可辨识量

以下参数是从 HF 轮地力数据中辨识得到的。

### 1. 每轮力偏置

每个轮子都有：

```text
b_Fx_i
b_Fy_i
b_Fz_i
```

含义：

```text
b_Fx_i: 第 i 个轮子的平均 Fx
b_Fy_i: 第 i 个轮子的平均 Fy
b_Fz_i: 第 i 个轮子的静态支撑载荷基线
```

其中 `b_Fz_i` 是最主要的项，表示每个轮子平均承担的垂向载荷。

输出形状：

```text
force_bias: [6, 3]
```

最后一维含义为：

```text
[Fx, Fy, Fz]
```

### 2. Fz 载荷转移系数

每个轮子都有：

```text
k_roll_i
k_pitch_i
k_ax_i
k_ay_i
k_az_i
```

含义：

```text
k_roll_i  : 车体 roll 对第 i 个轮子 Fz 的影响
k_pitch_i : 车体 pitch 对第 i 个轮子 Fz 的影响
k_ax_i    : 车体 acc_x 对第 i 个轮子 Fz 的影响
k_ay_i    : 车体 acc_y 对第 i 个轮子 Fz 的影响
k_az_i    : 车体 acc_z 对第 i 个轮子 Fz 的影响
```

这些参数描述车体姿态和加速度如何在六个轮子之间重新分配垂向载荷。

输出形状：

```text
fz_load: [6, 5]
```

最后一维含义为：

```text
[roll, pitch, acc_x, acc_y, acc_z]
```

### 3. 垂向速度阻尼系数

每个轮子都有：

```text
k_vz_i
```

它乘在：

```text
relu(-wheel_vz_i)
```

含义是轮子向下运动时的简单压缩速度项：

```text
wheel_vz_i < 0  ->  relu(-wheel_vz_i) > 0
wheel_vz_i >= 0 ->  relu(-wheel_vz_i) = 0
```

输出形状：

```text
fz_down_v: [6]
```

该项是可辨识的，但当前结果显示，主要改善来自 `roll/pitch/acc` 载荷转移项，而不是单独来自该速度阻尼项。

## 非选用但可诊断的量

验证脚本中还保留了以下可辨识诊断项：

```text
alpha_bekker * Bekker_force
Fx/Fy velocity/omega 经验项
```

这些量可以被脚本辨识，但当前消融结果显示它们不是主要有效项：

```text
load_transfer_only:
  vector_rmse = 84.63

load_transfer + Bekker:
  vector_rmse = 85.76

load_transfer + Fx/Fy empirical:
  vector_rmse = 84.61
```

解释：

```text
Bekker 项会让结果变差。
Fx/Fy 经验项只有极小改善。
```

因此它们可以保留为诊断选项，但不应作为当前选用模型的主计算路径。

## 固定结构

以下内容不是自由辨识参数，而是模型结构或脚本设定：

```text
wheel_id 与轮子的对应关系
输入列的选择规则
relu(-wheel_vz_i) 这一非线性形式
模型拓扑结构
```

这些由验证脚本固定。

## 物理解释

该模型说明，当前 HF 垂向力主要由以下部分解释：

```text
每轮静态支撑载荷
+ 车体姿态和加速度造成的载荷转移
```

当前结果不支持以下假设作为主导机制：

```text
sinkage -> Bekker 压强 -> 轮地力
```

也就是说，在这批 HF 数据上，Bekker 可以作为诊断特征保留，但不应被当成 HF force 的主要计算模型。

## 输出文件

辨识参数输出到：

```text
results_v3/diagnostics/<run_name>/fitted_gated_bekker_load_transfer.json
```

当前选用的 load-transfer-only 结果位于：

```text
results_v3/diagnostics/hf_fz_load_transfer_only/fitted_gated_bekker_load_transfer.json
```

验证指标输出到：

```text
results_v3/diagnostics/<run_name>/flat_phy_vs_hf_metrics.json
```

