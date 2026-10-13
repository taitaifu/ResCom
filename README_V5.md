# ResCom V5

V5 将状态重建和接触力预测改为级联训练：先从 LF 序列重建 Body、Rocker、Wheel 状态，再用预测状态计算 load-transfer 物理力，最后由轻量轮地接触历史编码器和共享 ForceResidualHead 预测剩余力误差。V5 使用独立代码、checkpoint 和 `results_v5/` 输出目录，不覆盖 V4。仓库环境及数据集准备说明见 [README.md](README.md)。

所有命令从仓库根目录运行：

```bash
cd /home/user/ResCom
```

## 代码入口

- `train/train_graph_model_v5.py`：`state_teacher`、`state_student`、`force`、`joint` 四个训练阶段。
- `models/graph_temporal_hgt_compensation_v5.py`：V5 State 网络、连续 Body rollout、状态重建、Contact History Encoder 和 ForceResidualHead。
- `models/data_utils_v5.py`：V5 特征列、case 数据集、连续 rollout window 和 contact history 数据。
- `models/load_transfer_force_v5.py`：由预测状态计算 Fz 物理基线，并在训练 split 上拟合每轮参数。
- `train_pipeline.sh`：运行 V5 状态到力的级联训练流程。

当前这组 V5 文件提供训练入口，没有独立的 V5 推理脚本。V4 推理入口和 checkpoint 不应当作 V5 推理接口使用。

## 数据流

```text
LF sequence
    ↓
Node encoders → HGT ×3 → 5-group readout → TCN → Teacher BiLSTM / Student causal LSTM
    ↓
Body / Rocker / Wheel corrected state
    ↓
predicted roll, pitch, body acceleration, wheel vertical velocity
    ↓
load-transfer Fphy_z
    ↓
wheel contact history encoder + corrected state features + force baseline
    ↓
shared ForceResidualHead → delta_Fxyz
    ↓
Ffinal = [LF_Fx, LF_Fy, Fphy_z] + delta_F
```

Teacher 和 Student 共用 V5 状态结构。Teacher 的 BiLSTM 可使用配置允许的未来 LF；Student 的因果时序网络只读取历史和当前 LF。Student 阶段保留 Teacher latent distillation。Force 阶段只从冻结的 Student checkpoint 生成 corrected state，不会用 LF 加速度或 LF wheel vertical velocity 直接替代预测状态的物理量。

### 状态重建

Body 使用两个独立输出头：`body_velocity_head` 预测 `delta_v`，`body_position_correction_head` 预测 `delta_p`。定义保持为：

```text
v_pred(t) = v_LF(t) + delta_v(t)
p_prop(t) = p_pred(t-1) + 0.5 * [v_pred(t-1) + v_pred(t)] * dt
p_pred(t) = p_prop(t) + gate_body_pos(t) * delta_p(t)
```

训练数据在每个 sample 内提供同一 case 的连续 rollout window。默认 `body_rollout_len=5`；每段从该 sample 的 LF body position/velocity 初始化，corrected position/velocity 只在当前 sample 的 rollout 步之间传递，不跨 batch 或 case 延续。`body_delta_smooth` 对相邻连续步的 `delta_p` 差值计算 Huber loss，默认权重 `0.01`。

Rocker 与 Wheel 位置以 `LF part position + body_global_delta` 为 base，位置 gate 只作用于经过预测 Body rotation 转换的局部 delta。局部位置监督先从 HF 位置差中减去真实 Body global delta，再用 HF Body rotation 转回 Body 坐标系。

### 物理力与 Force residual

每轮 Fz 物理模型输入预测的 roll/pitch、Body 三轴加速度和 wheel vertical velocity：

```text
Fphy_z = b_z + k_roll*roll + k_pitch*pitch
       + k_ax*ax + k_ay*ay + k_az*az
       + k_vz*max(-wheel_vz, 0)
```

`Fx` 和 `Fy` 的 baseline 是对应 LF 力，`Fz` 的 baseline 是 `Fphy_z`。共享 WheelContactHistoryEncoder 使用接触状态、sinkage、slip 和相对速度等局部轮地历史特征；共享 ForceResidualHead 输出三轴 `delta_F`。V5 不使用 Force gate 或第二个 Force Corrector。

load-transfer 参数由训练 split 的 HF 状态和力拟合，并保存在 checkpoint；validation/test 不参与参数拟合。

## 训练阶段

| 阶段 | 输入与训练内容 | 必需 checkpoint |
| --- | --- | --- |
| `state_teacher` | Teacher 状态网络和状态 loss；不执行 Force 分支 | 无 |
| `state_student` | 因果 Student 状态网络及 Teacher latent distillation；不执行 Force 分支 | `--init_ckpt` 指向 Teacher |
| `force` | 冻结 State 网络，训练 Contact History Encoder 和 ForceResidualHead | `--state_checkpoint` 指向 Student |
| `joint` | 从 Force checkpoint 初始化，只微调指定 State 子层和 Force 模块 | `--force_checkpoint` |

State 阶段的 checkpoint 包括 `best_state_total.pt`、`best_state_output.pt` 和 `best_rollout.pt`。Force 阶段保存 `best_force.pt`，Joint 阶段保存 `best_joint.pt`。每个 run 还保存 V5 column spec、scaler 和 checkpoint 中的架构配置；Force/Joint checkpoint 同时保存 load-transfer 参数。

## 推荐训练流程

默认 pipeline 按顺序执行 `state_teacher → state_student → force`。Joint 默认关闭；设置 `V5_RUN_JOINT=1` 才会继续运行。

```bash
bash train_pipeline.sh
```

常用环境变量可在启动前覆盖：

```bash
DEVICE=cuda:0 \
V5_SAVE_DIR=results_v5 \
V5_BATCH_SIZE=64 \
V5_GRAD_ACCUM_STEPS=1 \
V5_EPOCHS=100 \
V5_BODY_ROLLOUT_LEN=5 \
V5_NUM_WORKERS=4 \
bash train_pipeline.sh
```

## 分阶段手动运行

默认训练输出写入 `results_v5/<stage>/<timestamp>/`。后续阶段需要从对应 run 目录取 checkpoint。

### State Teacher

```bash
python train/train_graph_model_v5.py \
  --train_stage state_teacher \
  --feature_dir Feature_Selection/DataSet \
  --merged_csv Feature_Selection/DataSet/merged_error_dataset.csv \
  --save_dir results_v5
```

### State Student

```bash
python train/train_graph_model_v5.py \
  --train_stage state_student \
  --init_ckpt results_v5/state_teacher/<run>/best_state_total.pt \
  --feature_dir Feature_Selection/DataSet \
  --merged_csv Feature_Selection/DataSet/merged_error_dataset.csv \
  --save_dir results_v5
```

### Force

优先用 Student 的 `best_rollout.pt` 初始化 Force 阶段；若该文件不存在，可选择 `best_state_output.pt` 或 `best_state_total.pt`。

```bash
python train/train_graph_model_v5.py \
  --train_stage force \
  --state_checkpoint results_v5/state_student/<run>/best_rollout.pt \
  --feature_dir Feature_Selection/DataSet \
  --merged_csv Feature_Selection/DataSet/merged_error_dataset.csv \
  --save_dir results_v5
```

### 可选 Joint

```bash
python train/train_graph_model_v5.py \
  --train_stage joint \
  --force_checkpoint results_v5/force/<run>/best_force.pt \
  --feature_dir Feature_Selection/DataSet \
  --merged_csv Feature_Selection/DataSet/merged_error_dataset.csv \
  --save_dir results_v5
```

## 常用参数

- `--history_len`：历史输入长度，默认 `9`。
- `--teacher_future_len`：Teacher 可见的未来 LF 长度，默认 `3`；Student 输入仍保持因果。
- `--body_rollout_len`：sample 内连续 Body rollout 长度，默认 `5`。
- `--lambda_body_delta_smooth`：Body position correction 的时间平滑权重，默认 `0.01` 且必须为正数。
- `--state_checkpoint`：Force 阶段所用冻结 State checkpoint。
- `--force_checkpoint`：Joint 阶段初始化 checkpoint。
- `--state_trainable_layers`：Joint 阶段允许接收 Force gradient 的 State 网络层，默认包含最后一层 HGT 和 temporal 模块。
- `--batch_size`、`--grad_accum_steps`：microbatch 大小与梯度累积步数，默认 `64 × 1`，有效 batch 为 `64`。可通过 `V5_BATCH_SIZE` 和 `V5_GRAD_ACCUM_STEPS` 调整 Pipeline。
- `--epochs`、`--lr`、`--num_workers`：轮数、学习率和数据 worker 数。

查看全部参数：

```bash
python train/train_graph_model_v5.py --help
```

## 输出目录与 V4 隔离

V5 默认写入 `results_v5/`，并在 checkpoint 元数据中记录 `v5_role`、`train_stage`、状态/力网络架构参数、load-transfer 参数、scaler、column spec、`history_len`、`teacher_future_len` 和 `body_rollout_len`。V5 的模型、数据、物理力模块和训练入口均使用独立的 `_v5` 文件；执行 V5 pipeline 不会改写 V4 代码或 `results_v4/`。
