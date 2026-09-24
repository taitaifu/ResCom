# ResCom V4

V4 是用于从 LF 仿真状态重建 HF 车体、摇臂、车轮状态与轮地接触力的图时序补偿模型。相较 V3，V4 更新了 Force 路径，并在训练中使用 sample 内连续 Body rollout。仓库环境与数据集生成的通用说明见 [README.md](README.md)。

所有命令均从仓库根目录运行：

```bash
cd /home/user/ResCom
```

## 代码入口

- `train/train_graph_model_v4.py`：Teacher / Student 两阶段训练。
- `infer/infer_graph_model_v4.py`：merged CSV 或 Custom case 推理。
- `models/graph_temporal_hgt_compensation_v4.py`：V4 HGT、时序网络、位置/姿态/力预测头和可选 Force TCN。
- `models/data_utils_v4.py`：V4 特征分组、case 数据集、rollout 连续序列及 load-transfer 输入。
- `models/load_transfer_force_v4.py`：load-transfer 物理力计算。

## 模型概览

V4 将控制、Body、Rocker、Wheel kinematics 和 Wheel contact 特征编码为异构图节点，经 HGT 与 5-group readout 汇总，再通过 TCN 和角色对应的 LSTM。模型预测 body 状态残差、part 局部位置 delta、姿态变化与 wheel 接触力补偿。

Teacher 使用双向 LSTM，可使用目标时刻之后的 LF 输入。Student 使用因果 LSTM，只读取历史和当前 LF。Student 阶段加载冻结的 Teacher，并蒸馏 Teacher 的潜在表示。

默认启用 LF-only Force TCN，为轮地接触力分支提供时序特征；该路径可通过 `--no-use_force_tcn` 关闭。V4 的力学先验使用 load-transfer 力计算，与 V3 的 differentiable terramechanics 路径不同。

### Body rollout

训练默认 `--body_rollout_len 5`。每个样本包含同一 case 内连续的 rollout 输入、LF body state、HF target 和逐步 `dt`。rollout 从锚点 LF 状态开始，以预测的 corrected velocity 连续积分位置，再由现有 body residual head 的 position delta 做 gated correction。时间平滑项作用于同一样本相邻 rollout 步的 delta；不会跨 batch 或 case 传递状态。

每步单独构造输入窗口：Teacher 窗口可包含其配置的 future LF；Student 窗口截止于该步当前时刻，不读取未来 LF。Rocker / Wheel rollout 位置监督使用预测的 Body global delta，并对 body-frame local delta 做监督。

## 数据要求

默认数据位置为：

```text
Feature_Selection/DataSet/
├── merged_error_dataset.csv
└── 特征列配置文件
```

CSV 按 `case_name` 和 `time` 组织，需包含 V4 列配置定义的 LF 输入，以及 HF body、wheel kinematics、wheel contact 等监督列。case 应按时间顺序排列，且 rollout 区间内不能跨 case；数据按 case 切分 train/validation/test。`pred_horizon` 必须为 `0`。

训练默认 `history_len=9`、`teacher_future_len=3`、`body_rollout_len=5`。每个 rollout sample 需要有完整连续窗口，因此每个 case 尾部不足 rollout 或 Teacher future 窗口的样本不会进入训练集。DataLoader 可以 shuffle，因为递推状态仅在 sample 内维护。

## 训练

### Teacher

```bash
python3 train/train_graph_model_v4.py \
  --feature_dir /home/user/ResCom/Feature_Selection/DataSet \
  --merged_csv /home/user/ResCom/Feature_Selection/DataSet/merged_error_dataset.csv \
  --save_dir /home/user/ResCom/results_v4 \
  --train_stage teacher
```

### Student

将 `<teacher-run>` 替换为 Teacher 的实际时间戳目录。Student 也会在 `--save_dir/teacher/` 下自动查找 checkpoint；显式指定 checkpoint 可以固定初始化版本。

```bash
python3 train/train_graph_model_v4.py \
  --feature_dir /home/user/ResCom/Feature_Selection/DataSet \
  --merged_csv /home/user/ResCom/Feature_Selection/DataSet/merged_error_dataset.csv \
  --save_dir /home/user/ResCom/results_v4 \
  --train_stage student \
  --init_ckpt /home/user/ResCom/results_v4/teacher/<teacher-run>/best_safe.pt
```

常用参数：

- `--history_len`、`--teacher_future_len`：时序输入窗口。
- `--body_rollout_len`：Body 连续 rollout 步数，默认 5，必须大于等于 1。
- `--lambda_body_delta_smooth`：相邻 rollout 步 delta 平滑损失权重，默认 0。
- `--kinematic_pos_gate_scale`、`--lambda_kin`：位置 gate 缩放和运动学一致性权重。
- `--use_force_tcn` / `--no-use_force_tcn`：启用或关闭 LF-only Force TCN。
- `--batch_size`、`--epochs`、`--lr`、`--hidden_dim`、`--graph_layers`、`--tcn_dim`、`--lstm_dim`、`--num_workers`：训练规模与网络宽度参数。

完整参数可查看 `python3 train/train_graph_model_v4.py --help`。默认训练 120 epochs、batch size 512。训练结果写入 `<save_dir>/<teacher|student>/<timestamp>/`，包括 `best_total.pt`、`best_output.pt`、`best_safe.pt`、`history.json`、V4 列配置、scaler 和 TensorBoard 日志。Student 自动查找 checkpoint 时优先使用 `best_output.pt`，然后尝试 `best_safe.pt`、`best_total.pt` 和 `best_model.pt`。

## 推理

### 使用 merged CSV

```bash
python3 infer/infer_graph_model_v4.py \
  --checkpoint /home/user/ResCom/results_v4/teacher/<teacher-run>/best_safe.pt \
  --feature_dir /home/user/ResCom/Feature_Selection/DataSet \
  --merged_csv /home/user/ResCom/Feature_Selection/DataSet/merged_error_dataset.csv \
  --infer_mode merged_csv \
  --split test
```

### 推理单个 Custom case

```bash
python3 infer/infer_graph_model_v4.py \
  --checkpoint /home/user/ResCom/results_v4/student/<student-run>/best_safe.pt \
  --feature_dir /home/user/ResCom/Feature_Selection/DataSet \
  --infer_mode custom_case \
  --custom_case_dir /path/to/Zhurong_Custom_case0238 \
  --sph_case_dir /path/to/Zhurong_SPH_case0238 \
  --infer_case_name case0238
```

`--sph_case_dir` 可选，用于提供参考 HF case。批量 Custom 推理使用 `--infer_mode custom_cases`，可通过 `--custom_cases_dir` 和 `--sph_cases_dir` 指定目录；默认模式为 `custom_cases`。`--model_role auto`（默认）从 checkpoint 判断 Teacher 或 Student，也可显式指定角色。

推理输出默认保存在 checkpoint 目录下新建的 `infer_<case>_<timestamp>/`，包括 `predictions_v4.csv`、`metrics_v4.json`、`run_info.txt` 和 `plots/`。Custom 输入还会保存生成的推理输入 CSV。推理脚本对各目标 sample 单独执行 forward；训练用的 5 步 rollout 参数不改变推理入口的 sample 处理方式。

## 常用推理选项

- `--split all|train|val|test`：merged CSV split，默认 `test`。
- `--case_name <name>`：选择特定 case。
- `--history_len`、`--teacher_future_len`：覆盖 checkpoint 中的窗口参数。
- `--kinematic_pos_gate_scale`：覆盖 checkpoint 保存的位置 gate 缩放。
- `--batch_size`、`--num_workers`、`--device`：推理资源选项。
- `--output_dir`：覆盖默认输出目录。

