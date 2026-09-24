# Graph Temporal HGT Compensation V2 说明

## 新增文件

- `models/graph_temporal_hgt_compensation_v2.py`：V2 HGT + TCN + BiLSTM 模型。
- `models/data_utils_v2.py`：V2 数据列分组、数据集封装和兼容读取逻辑。
- `models/differentiable_terramechanics.py`：轻量可微地面力学模块，全部使用 PyTorch Tensor。
- `train/train_graph_model_v2.py`：V2 Teacher / Student 两阶段训练入口。
- `train_graph_model_v2.py`：仓库根目录下的便捷启动入口。
- `models/graph_temporal_compensation.py`：兼容当前 `models/__init__.py` 导入的薄封装。

V2 不修改原始 `models/graph_temporal_hgt_compensation.py`、`models/data_utils.py` 和 `train/train_graph_model.py`。

## 网络结构

V2 删除 `target_error_*` 节点和 `state_to_target` 关系。异构图只保留：

- control context
- body
- rocker 节点
- wheel kinematic 节点
- wheel contact 节点

模型默认保留 3 层 HGT，并保留 Rocker 节点、RelationGate、EdgeGate、TCN 和 BiLSTM。

HGT 输出不再把所有节点直接 flatten。每个时间步分别读取 5 组图特征：

- control
- body
- rocker mean
- wheel kinematic mean
- wheel contact mean

这 5 组特征拼接后经过 `Linear` 降维，再进入 TCN + BiLSTM。graph readout 和 RelationGate context 也使用同样的 5 组特征，不再额外重复 `observed_mean`。

V2 保持单步预测，删除 `horizon_embed`。

## Teacher / Student 两阶段

Teacher 阶段训练：

- 主体 HGT / TCN / BiLSTM
- `TeacherGateNet`
- `EdgeGate`
- 状态残差预测头
- `F_bias` 和 `F_dynamic` 两个接触力残差预测头

Student 阶段加载 Teacher 最优模型，冻结主体网络、`TeacherGateNet` 和 `EdgeGate`，主要训练 `StudentGateSummaryNet`。Student 根据 LF 可见信息预测 Teacher summary，然后把预测 summary 输入冻结的 `TeacherGateNet` 得到 RelationGate。

Teacher summary 只使用 LF + RES。HF 继续作为监督标签，但不再进入 Teacher summary。

## 预测目标

网络预测状态残差，并用 LF 当前状态重构补偿状态：

- body：`pos_x/y/z`、`vel_x/y/z`
- 每个车轮：`pos_x/y/z`、车轮角速度 `omega` 或数据中可匹配的 `ang_vel_*`
- contact：仅 `Fx/Fy/Fz`

车轮不额外预测线速度。`Mx/My/Mz` 不再作为输出、监督或评价指标，但仍可作为 wheel contact 输入特征。

### 车轮局部位置补偿

当前 V2 中 body head 继续预测车身位置残差 `delta_body_pos`。wheel head 的位置通道不再表示完整车轮位置残差，而表示局部车轮补偿：

```text
delta_wheel_local =
(HF wheel position - LF wheel position)
- (HF body position - LF body position)

wheel_pos_pred =
wheel_pos_LF
+ delta_body_pos
+ delta_wheel_local
```

其中 `delta_body_pos` 是六轮共享的整体平移修正，不重新计算正运动学。wheel head 输入额外拼接对应 rocker hidden feature：

```text
wheel0 -> lf
wheel1 -> rf
wheel2 -> lb
wheel3 -> rb
wheel4 -> lb
wheel5 -> rb
```

wheel head 输入为：

```text
temporal feature + body feature + corresponding rocker feature + wheel_i feature + global readout
```

## 可微地面力学模块

`models/differentiable_terramechanics.py` 参考 `models/TerrainPlugins/TerramechanicsRigid.cpp` 中的核心公式，实现轻量 PyTorch 版本：

- `CalculateSlip`
- `CalculateBeta`
- `CalculationTheta1`
- `WheelTerrainInteraction` 中计算 `Fx/Fy/Fz` 所需的主要关系

该模块不重建完整地形和接触流程，不执行：

- `TerrainGridMap` 地形搜索
- `CalculateTouchArea`
- 接触区域重建
- 接触面重新拟合
- `ApplyToBody`
- `Mx/My/Mz` 计算或预测

V2 使用数据中已有 LF 接触状态和地形/轮地力学参数为基准，重新计算：

```text
slip_pred = CalculateSlip(omega_pred, v_pred)
beta_pred = CalculateBeta(v_pred)
pa/pb/pc = TerrainGridMap.SamplePoint(contact patch samples)
contact_plane = fit_plane(pa, pb, pc)
sinkage_pred = clamp(r - dot(contact_normal, wheel_pos_pred - contact_point) / |contact_normal|, 0, sinkage_max)
F_phy = [Fx_phy, Fy_phy, Fz_phy]
```

当前实现按 `TerramechanicsRigid::CalculateTouchArea` 的三点采样思想，从 `zero_terrain_reader.txt` 的地形网格中估计接触平面的 `contact_point/contact_normal`，再用预测车轮位置到接触平面的法向距离计算沉陷量。只有 batch 中没有接触平面信息时，才回退到直接地形高度或旧的 LF sinkage 修正逻辑。

所有计算均使用 PyTorch Tensor，不使用 NumPy 或外部 C++ 调用。训练中对梯度边界做了区分：`L_dyn` 保留 `F_phy -> slip / sinkage -> body velocity / wheel position / wheel omega` 的弱耦合；`L_force/L_bias/L_dynamic` 使用 detach 后的状态或离线 `F_phy_ref`，避免接触力误差反向破坏状态 head。

实现中对反向传播做了数值安全处理：`acos` 输入会避开 `[-1, 1]` 边界，`pow` 底数会避开 0，`tan(phi)` 会限制角度范围，除法和指数项使用安全分母和指数裁剪。这些处理用于避免地面力学公式出现“forward 有限但 backward 产生 NaN/Inf”的情况。

地形参数默认从 `models/TerrainPlugins/zero_terrain_reader.txt` 读取。该文件列顺序为：

```text
x y z Kc Kphi n0 n1 c phi K
```

读取时会先判断 `Kc/Kphi/n0/n1/c/phi/K` 是否随位置变化。如果参数不变，则固定使用第一行参数；如果参数变化，则根据 LF 车轮 `x/y` 位置查找最近地形点。没有该文件时，默认使用第一行对应的硬编码参数：

```text
Kc=-20700, Kphi=1594800, n0=0.79, n1=0.70, c=460, phi=0.61, K=0.0133
```

地面力学指数按 `n = n0 + n1 * abs(slip)` 计算。

## 接触力三部分预测

每个车轮最终接触力为：

```text
F_pred = F_phy + F_bias + F_dynamic
```

- `F_phy`：由可微地面力学模型计算，描述补偿状态变化导致的物理接触力变化。
- `F_bias`：低频残差头，主要使用当前 HGT 空间/接触特征。
- `F_dynamic`：动态残差头，主要使用 TCN + BiLSTM 时序特征。

数据集构造阶段会用 HF 状态离线计算参考物理力：

```text
F_phy_ref = Physics(HF body velocity, HF wheel position, HF wheel omega)
F_res_ref = F_HF - F_phy_ref
F_bias_target = LPF(F_res_ref)
F_dynamic_target = F_res_ref - F_bias_target
```

## 损失项

V2 总损失：

```text
L_total =
L_state
+ lambda_wheel_local * L_wheel_local
+ lambda_assembly * L_assembly
+ lambda_wheel_local_reg * L_wheel_local_reg
+ lambda_F * L_force
+ lambda_phy_state * L_phy_state
+ lambda_bias * L_bias
+ lambda_dynamic * L_dynamic
+ lambda_kin * L_kin
+ lambda_dyn * L_dyn
+ lambda_reg * L_residual_reg
+ lambda_harm * L_no_harm
```

各项含义：

- `L_state`：body p/v + wheel p/omega。
- `L_wheel_local`：监督 wheel head 输出的 `delta_wheel_local`。
- `L_assembly`：约束 `wheel_pos_pred - body_pos_pred` 接近 HF 中的车轮相对车身位置。
- `L_wheel_local_reg`：对局部车轮补偿做弱 L2 正则，抑制 body head 和 wheel head 大幅相互抵消。
- `L_force`：最终 `Fx/Fy/Fz` 监督。
- `L_phy_state`：使用非 detach 预测状态计算 `F_phy_pred`，监督其接近离线 `F_phy_ref`；默认 `lambda_phy_state=0.005`。
- `L_bias`：监督 `F_bias` 接近完整轨迹低通后的 `F_bias_target`。
- `L_dynamic`：监督 `F_dynamic` 接近完整轨迹高频残差 `F_dynamic_target`。
- `L_kin`：body 单步位置速度运动学一致性。
- `L_dyn`：body 平动加速度与六轮合力一致性。
- `L_residual_reg`：按力标准差标准化后的 `(F_bias/sigma_F)^2 + (F_dynamic/sigma_F)^2` 弱正则。
- `L_no_harm`：逐状态逐方向 no-harm 惩罚。

当前默认权重更偏向 bias residual 分支拟合：`lambda_phy_state=0.005`、`lambda_bias=0.6`、`lambda_dynamic=0.35`、`lambda_dyn=0.005`、`lambda_reg=5e-5`。

位置、速度、力和加速度均按 xyz 三方向分别计算，再在对应损失内部汇总；不把 xyz 向量模长作为唯一监督。接触力相关损失在有效 contact mask 下计算。

wheel 相关损失使用 `delta_body_pos.detach()` 构造车轮位置，因此 wheel position loss、`L_wheel_local`、`L_assembly` 和 wheel no-harm 不会反向修改 body head。接触力监督使用 detach 后的 body velocity、wheel position 和 wheel omega 计算 `F_phy_force`，因此 `L_force` 不会通过地面力学链回传到状态 head。`L_phy_state` 使用非 detach 的 `F_phy_pred` 对齐 `F_phy_ref`，`L_dyn` 使用非 detach 的预测状态和最终接触力，保留状态分支与接触力分支之间的弱物理一致性。

`F_phy`、`F_bias` 和 `F_dynamic` 本身保持物理单位 N，便于解释和输出；但 `L_force/L_bias/L_dynamic/L_phy_state` 会按对应 `Fx/Fy/Fz` 的训练集标准差 `sigma_F` 标准化后计算，避免力的量纲直接压过状态损失。`F_final_rmse` 和各物理力 RMSE 仍以原始力单位统计。

## No-Harm 惩罚

V2 不用 overall RMSE 判断是否恶化，而是逐通道计算：

```text
e_pred_k = abs(pred_k - HF_k) / sigma_k
e_LF_k   = abs(LF_k - HF_k) / sigma_k
L_no_harm_k = ReLU(e_pred_k - e_LF_k)
```

当前实现覆盖：

- body pos xyz
- body vel xyz
- wheel pos xyz
- wheel omega
- contact Fx/Fy/Fz

训练日志中同时输出 body、wheel 和 contact 的 no-harm violation rate。

## 运动学和动力学一致性

单步运动学一致性：

```text
p_pred_t ~= p_HF_(t-1) + 0.5 * (v_HF_(t-1) + v_pred_t) * dt
```

动力学一致性：

```text
a_kin = (v_pred_t - v_HF_(t-1)) / dt
a_force = (sum(F_pred_world) + gravity + other_known_external_force) / mass
```

当前 V2 只做平动动力学，不加入力矩和角动力学损失。六轮预测力按同一坐标系直接求和；如后续数据提供更明确的轮地接触坐标系，可在 `compute_losses_v2` 中加入显式坐标变换。

## 训练命令

默认 V2 结果保存根目录为：

```text
/home/user/ResCom/results/teacher_fullv2
```

Teacher 训练：

```bash
python train_graph_model_v2.py --train_stage teacher
```

默认输出目录形式：

```text
/home/user/ResCom/results/teacher_fullv2/teacher/<run_time>/
```

Student 训练：

```bash
python train_graph_model_v2.py --train_stage student --init_ckpt /home/user/ResCom/results/teacher_fullv2/teacher/<run_time>/best_model.pt
```

如果不提供 `--init_ckpt`，Student 会尝试从默认保存根目录下自动寻找最新 Teacher checkpoint。

TensorBoard 日志默认写入当前 run 的 `tensorboard/` 子目录：

```text
/home/user/ResCom/results/teacher_fullv2/<stage>/<run_time>/tensorboard/
```

查看日志：

```bash
tensorboard --logdir /home/user/ResCom/results/teacher_fullv2
```

如需指定单独日志目录：

```bash
python train_graph_model_v2.py --train_stage teacher --log_dir /tmp/rescom_v2_tb
```

训练过程默认使用 `tqdm` 进度条显示 batch 进度，并在进度条上刷新主要 loss 组成。刷新频率由 `--progress_interval` 控制：

```bash
python train_graph_model_v2.py --train_stage teacher --progress_interval 10
```

如需关闭进度条并退回普通文本输出：

```bash
python train_graph_model_v2.py --train_stage teacher --no-progress_bar
```

## 主要输出指标

V2 训练脚本至少记录：

- body pos xyz RMSE
- body vel xyz RMSE
- wheel local correction xyz RMSE
- wheel force `Fx/Fy/Fz` RMSE
- `F_phy` RMSE
- final force RMSE
- 所有独立 loss
- `L_wheel_local`、`L_assembly`、`L_wheel_local_reg`
- body correction xyz 幅值
- wheel local correction xyz 幅值
- Student latent / gate 蒸馏 loss
- body pos/vel、wheel pos/omega、contact force 的 no-harm violation rate
- RelationGate / EdgeGate 输出张量，供后续日志统计扩展

这些指标会同时保存到 `history.json`，并写入 TensorBoard 的 `train/*` 与 `val/*` 标量曲线。
每个 epoch 结束时，终端也会分别打印 train / val 的主要力相关 loss、`F_final_rmse/F_phy_rmse`、wheel0 三方向力 RMSE，以及 LF sinkage 和修正后 sinkage 统计。

## 数值异常定位

如果训练后期出现只剩少数指标打印，通常表示部分 loss 或输出已经变成 `NaN/Inf`。可以启用 batch 级 finite check：

```bash
python train_graph_model_v2.py --train_stage teacher --debug_finite
```

开启后，每个 batch 会检查：

- `losses` 中的关键 loss 和 RMSE
- 模型 `output`
- backward 后的参数梯度
- gradient clipping 前的总体梯度范数

一旦发现第一个 `NaN/Inf`，训练会立即停止，并打印：

- epoch
- train / val 阶段
- batch 编号
- batch 中前几个 `case_name`
- batch 时间范围
- 出问题的 tensor 名称、shape、NaN/Inf 数量和有限值统计

异常 batch 会保存到：

```text
<当前run目录>/debug_finite/
```

也可以手动指定保存目录：

```bash
python train_graph_model_v2.py --train_stage teacher --debug_finite --debug_dump_dir /tmp/rescom_v2_debug
```

## 维度和 DataLoader 建议

常用维度建议：

- `--hidden_dim 96` 或 `128`：HGT 节点隐藏维度。数据量较大时优先用 `128`，显存紧张时用 `96`。
- `--tcn_dim 128` 或 `192`：TCN 隐藏维度。默认 `192` 更适合保留动态信息。
- `--lstm_dim 128`：BiLSTM 输出维度。通常不建议一开始调得过大。
- `--lstm_layers 1` 或 `2`：训练慢或过拟合时用 `1`，默认用 `2`。
- `--graph_layers 3`：V2 默认 3 层 HGT，建议保持。
- `--batch_size 64` 或 `128`：9406 个 batch 这种情况说明滑窗样本很多；显存允许时可增大 batch size 减少每个 epoch 的 batch 数。

`--num_workers` 是 PyTorch `DataLoader` 的并行取样进程数：

- `0`：主进程取数据，最稳定，但大数据集可能慢。
- `4` 到 `8`：并行准备 batch，通常能减少 GPU 等数据的时间。
- 过大可能导致 CPU、内存或磁盘争用，反而变慢。

建议先试：

```bash
python train_graph_model_v2.py --train_stage teacher --num_workers 6 --batch_size 128
```

## 与原版本的主要区别

- 删除 `target_error_*` 节点。
- 删除 `state_to_target` 关系。
- 删除 `horizon_embed`。
- HGT 输出不再全节点 flatten 后进入 TCN。
- Teacher summary 删除 HF，只保留 LF + RES。
- Student 直接复用冻结的 `TeacherGateNet`，不再使用 `student_summary_gate_net`。
- EdgeGate 在 Teacher 阶段训练，Student 阶段冻结复用。
- contact 输出仅 `Fx/Fy/Fz`。
- 接触力改为 `F_phy + F_bias + F_dynamic`。
- 删除 `Mx/My/Mz` 预测、监督和评价。
- 新增可微地面力学、基于 `F_phy_ref` 的 bias/dynamic 分解监督、No-Harm、单步运动学一致性和平动动力学一致性。
