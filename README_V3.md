# ResCom V3

V3 是一套用于从低保真（LF）仿真状态重建高保真（HF）车体、摇臂、车轮状态与轮地接触力的图时序补偿模型。本文说明 V3 的训练、checkpoint 和推理入口。仓库环境与数据集生成的通用说明见 [README.md](README.md)。

所有命令均从仓库根目录运行：

```bash
cd /home/user/ResCom
```

## 代码入口

- `train/train_graph_model_v3.py`：Teacher / Student 两阶段训练。
- `infer/infer_graph_model_v3.py`：merged CSV 或 Custom case 推理。
- `models/graph_temporal_hgt_compensation_v3.py`：HGT、时序网络和状态/力预测头。
- `models/data_utils_v3.py`：特征分组、按 case 切分的数据集和 V3 batch 构造。
- `models/differentiable_terramechanics.py`：可微轮地力学计算，用于接触力相关预测和约束。

## 模型概览

V3 将 system/control、body、rocker、wheel kinematics 和 wheel contact 特征组织为异构图节点，经 HGT 编码后按 5 组 readout 汇总，再经过 TCN 与角色对应的 LSTM。预测头重建 Body、Rocker、Wheel 状态和轮地接触力；训练目标还包含运动学、装配关系、接触力学及 no-harm 等项。

Teacher 使用双向时序网络，并可读取目标时刻之后的 LF 窗口。Student 使用因果时序网络，只读取历史和当前 LF。Student 训练时加载并冻结 Teacher 主体，通过潜在表示蒸馏学习 Teacher 表示。

V3 的接触力路径结合可微 terramechanics 与残差补偿；与 V4 的 LF-only Force TCN 路径不同。

## 数据要求

默认数据位置为：

```text
Feature_Selection/DataSet/
├── merged_error_dataset.csv
└── 特征列配置文件
```

CSV 按 `case_name` 和 `time` 组织，需包含 V3 列配置定义的 LF 输入及对应 HF body、wheel kinematics、wheel contact 监督列。训练集、验证集和测试集按 case 切分，避免同一 case 同时出现在多个 split。`pred_horizon` 必须为 `0`，即监督目标对应当前时刻。

训练和推理默认参数包括 `history_len=9`、`teacher_future_len=3`。Teacher 窗口使用历史、当前和未来 LF；Student 窗口仅使用历史和当前 LF。推理时 Teacher checkpoint 使用未来 LF，Student checkpoint 不使用未来 LF。

## 训练

### Teacher

```bash
python3 train/train_graph_model_v3.py \
  --feature_dir /home/user/ResCom/Feature_Selection/DataSet \
  --merged_csv /home/user/ResCom/Feature_Selection/DataSet/merged_error_dataset.csv \
  --save_dir /home/user/ResCom/results_v3 \
  --train_stage teacher
```

### Student

将 `<teacher-run>` 替换为 Teacher 的实际时间戳目录。Student 也可在 `--save_dir/teacher/` 下自动查找可用 Teacher checkpoint；显式指定可确保加载预期权重。

```bash
python3 train/train_graph_model_v3.py \
  --feature_dir /home/user/ResCom/Feature_Selection/DataSet \
  --merged_csv /home/user/ResCom/Feature_Selection/DataSet/merged_error_dataset.csv \
  --save_dir /home/user/ResCom/results_v3 \
  --train_stage student \
  --init_ckpt /home/user/ResCom/results_v3/teacher/<teacher-run>/best_safe.pt
```

默认训练配置为 120 epochs、batch size 512。常用可调参数有 `--history_len`、`--teacher_future_len`、`--batch_size`、`--epochs`、`--lr`、`--hidden_dim`、`--graph_layers`、`--tcn_dim`、`--lstm_dim` 和 `--num_workers`。运行 `python3 train/train_graph_model_v3.py --help` 查看完整列表。

每次运行保存到 `<save_dir>/<teacher|student>/<timestamp>/`，其中包括 `best_total.pt`、`best_output.pt`、`best_safe.pt`、`history.json`、V3 列配置、scaler 和 TensorBoard 日志。Student 自动查找 checkpoint 时优先选择 `best_output.pt`，其次是 `best_safe.pt`、`best_total.pt` 和 `best_model.pt`。

## 推理

### 使用 merged CSV

```bash
python3 infer/infer_graph_model_v3.py \
  --checkpoint /home/user/ResCom/results_v3/teacher/<teacher-run>/best_safe.pt \
  --feature_dir /home/user/ResCom/Feature_Selection/DataSet \
  --merged_csv /home/user/ResCom/Feature_Selection/DataSet/merged_error_dataset.csv \
  --infer_mode merged_csv \
  --split test
```

### 推理单个 Custom case

```bash
python3 infer/infer_graph_model_v3.py \
  --checkpoint /home/user/ResCom/results_v3/student/<student-run>/best_safe.pt \
  --feature_dir /home/user/ResCom/Feature_Selection/DataSet \
  --infer_mode custom_case \
  --custom_case_dir /path/to/Zhurong_Custom_case0238 \
  --sph_case_dir /path/to/Zhurong_SPH_case0238 \
  --infer_case_name case0238
```

`--sph_case_dir` 是可选参考数据。批量推理可使用 `--infer_mode custom_cases`，并通过 `--custom_cases_dir`、`--sph_cases_dir` 指定目录。推理模式还支持 `merged_csv`；默认模式为 `custom_cases`。`--model_role auto`（默认）从 checkpoint 判断 Teacher 或 Student，也可显式设为 `teacher` 或 `student`。

推理输出默认写入 checkpoint 目录下新建的 `infer_<case>_<timestamp>/`，包括 `predictions_v3.csv`、`metrics_v3.json`、`run_info.txt` 和 `plots/`。输入为 Custom case 时还会保存生成的推理输入 CSV。

## 常用推理选项

- `--split all|train|val|test`：merged CSV 使用的数据 split，默认 `test`。
- `--case_name <name>`：只选择指定 case。
- `--history_len`、`--teacher_future_len`：未从 checkpoint 参数恢复时覆盖窗口长度。
- `--batch_size`、`--num_workers`：推理批次和数据加载 worker 数。
- `--output_dir`：覆盖默认推理结果目录。
- `--device`：例如 `cpu` 或 `cuda:0`；未指定时由脚本选择。

