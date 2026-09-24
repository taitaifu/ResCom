# ResCom Runbook

本文档整理数据集生成、教师网络训练、学生网络训练和推理命令。所有命令默认从仓库根目录运行。

```bash
cd /home/user/ResCom
```

## 环境

如果需要创建/更新 Conda 环境：

```bash
conda env create -f ResComenv.yml
conda activate ResComenv
```

如果环境已存在：

```bash
conda activate ResComenv
```

## 生成训练数据集

默认输入路径：

- 低保真 Custom：`/media/user/新加卷/RoverSimData/Multi_Custom_output`
- 高保真 SPH：`/media/user/新加卷/RoverSimData/Multi_SPH_output`
- 输出目录：`/home/user/ResCom/Feature_Selection/error_dataset`

直接运行：

```bash
python3 Feature_Selection/datasetProcess.py
```

显式指定输入/输出路径：

```bash
python3 Feature_Selection/datasetProcess.py \
  --lf-root "/media/user/新加卷/RoverSimData/Multi_Custom_output" \
  --hf-root "/media/user/新加卷/RoverSimData/Multi_SPH_output" \
  --out-dir "/home/user/ResCom/Feature_Selection/error_dataset"
```

运行完成后应看到：

```text
完成
/home/user/ResCom/Feature_Selection/error_dataset/merged_error_dataset.csv
```

检查输出文件：

```bash
ls -lh /home/user/ResCom/Feature_Selection/error_dataset
```

应至少包含：

```text
merged_error_dataset.csv
base_feature_columns.csv
proxy_feature_columns.csv
target_columns.csv
res_columns.csv
invalid_check_log.txt
```

## 教师网络训练

使用原来的数据路径 `Feature_Selection/DataSet`：

```bash
python3 train/train_graph_model.py \
  --feature_dir /home/user/ResCom/Feature_Selection/DataSet \
  --merged_csv /home/user/ResCom/Feature_Selection/DataSet/merged_error_dataset.csv \
  --train_stage teacher_full \
  --save_dir /home/user/ResCom/results/
```

训练输出会保存到：

```text
/home/user/ResCom/results/teacher_full/<运行时间>/
```

最重要的 checkpoint：

```text
/home/user/ResCom/results/0708/teacher_full/<运行时间>/best_model.pt
```

## 学生网络训练

学生网络默认会从同一个 `--save_dir` 下自动寻找最新的教师网络 `best_model.pt`：

```bash
python3 train/train_graph_model.py \
  --feature_dir /home/user/ResCom/Feature_Selection/DataSet \
  --merged_csv /home/user/ResCom/Feature_Selection/DataSet/merged_error_dataset.csv \
  --train_stage student_full \
  --save_dir /home/user/ResCom/results/0708
```

也可以手动指定教师 checkpoint 初始化：

```bash
python3 train/train_graph_model.py \
  --feature_dir /home/user/ResCom/Feature_Selection/DataSet \
  --merged_csv /home/user/ResCom/Feature_Selection/DataSet/merged_error_dataset.csv \
  --train_stage student_full \
  --save_dir /home/user/ResCom/results/0708 \
  --init_ckpt /home/user/ResCom/results/0708/teacher_full/<运行时间>/best_model.pt
```

学生网络输出会保存到：

```text
/home/user/ResCom/results/0708/student_full/<运行时间>/
```

## 推理

### 使用 merged_error_dataset.csv 推理

教师网络 checkpoint 推理：

```bash
python3 infer/infer_graph_model.py \
  --checkpoint /home/user/ResCom/results/0708/teacher_full/<运行时间>/best_model.pt \
  --feature_dir /home/user/ResCom/Feature_Selection/DataSet \
  --merged_csv /home/user/ResCom/Feature_Selection/DataSet/merged_error_dataset.csv \
  --infer_mode merged_csv \
  --gate_mode teacher
```

学生网络 checkpoint 推理：

```bash
python3 infer/infer_graph_model.py \
  --checkpoint /home/user/ResCom/results/0708/student_full/<运行时间>/best_model.pt \
  --feature_dir /home/user/ResCom/Feature_Selection/DataSet \
  --merged_csv /home/user/ResCom/Feature_Selection/DataSet/merged_error_dataset.csv \
  --infer_mode merged_csv \
  --gate_mode student
```

指定只推理某个 case：

```bash
python3 infer/infer_graph_model.py \
  --checkpoint /home/user/ResCom/results/0708/teacher_full/<运行时间>/best_model.pt \
  --feature_dir /home/user/ResCom/Feature_Selection/DataSet \
  --merged_csv /home/user/ResCom/Feature_Selection/DataSet/merged_error_dataset.csv \
  --infer_mode merged_csv \
  --gate_mode teacher \
  --case_name case0238
```

推理结果默认保存到 checkpoint 同级目录下的：

```text
infer_<运行时间>/
```

主要输出：

```text
predictions.csv
metrics.json
run_info.txt
plots/
```

### 使用单个 Custom case 推理

只输入低保真 Custom case：

```bash
python3 infer/infer_graph_model.py \
  --checkpoint /home/user/ResCom/results/0708/teacher_full/<运行时间>/best_model.pt \
  --feature_dir /home/user/ResCom/Feature_Selection/DataSet \
  --infer_mode custom_case \
  --gate_mode teacher \
  --custom_case_dir "/media/user/新加卷/RoverSimData/Multi_Custom_output/Zhurong_Custom_case0238"
```

如果同时有对应 SPH case，可用于参考对比：

```bash
python3 infer/infer_graph_model.py \
  --checkpoint /home/user/ResCom/results/0708/teacher_full/<运行时间>/best_model.pt \
  --feature_dir /home/user/ResCom/Feature_Selection/DataSet \
  --infer_mode custom_case \
  --gate_mode teacher \
  --custom_case_dir "/media/user/新加卷/RoverSimData/Multi_Custom_output/Zhurong_Custom_case0238" \
  --sph_case_dir "/media/user/新加卷/RoverSimData/Multi_SPH_output/Zhurong_SPH_case0238" \
  --infer_case_name case0238
```

### 使用多个 Custom case 推理

批量推理默认从以下目录读取多组 case：

```text
/media/user/新加卷/RoverSimData/Multi_Custom_test_output
/media/user/新加卷/RoverSimData/Multi_SPH_test_output
```

目录下的 `Zhurong_Custom_test_caseXXXX` 会按 `caseXXXX` 自动匹配对应的 `Zhurong_SPH_test_caseXXXX`。扫描逻辑也兼容包含 `Custom` / `SPH` 和 `_case` 的其它同类命名。如果提供了 SPH 父目录但缺少对应 case，程序会直接报错，避免指标使用错误参考值。

```bash
python3 infer/infer_graph_model.py \
  --checkpoint /home/user/ResCom/results/0708/teacher_full/<运行时间>/best_model.pt \
  --feature_dir /home/user/ResCom/Feature_Selection/DataSet \
  --infer_mode custom_cases \
  --gate_mode teacher
```

也可以显式指定批量目录：

```bash
python3 infer/infer_graph_model.py \
  --checkpoint /home/user/ResCom/results/0708/teacher_full/<运行时间>/best_model.pt \
  --feature_dir /home/user/ResCom/Feature_Selection/DataSet \
  --infer_mode custom_cases \
  --gate_mode teacher \
  --custom_cases_dir "/media/user/新加卷/RoverSimData/Multi_Custom_test_output" \
  --sph_cases_dir "/media/user/新加卷/RoverSimData/Multi_SPH_test_output"
```

## 常用检查

检查数据集中是否存在离谱位置/残差：

```bash
python3 Feature_Selection/check_dataset_outliers.py \
  --csv /home/user/ResCom/Feature_Selection/DataSet/merged_error_dataset.csv \
  --out-dir /home/user/ResCom/Feature_Selection/DataSet/outlier_check
```

输出文件：

```text
/home/user/ResCom/Feature_Selection/DataSet/outlier_check/summary.json
/home/user/ResCom/Feature_Selection/DataSet/outlier_check/position_column_stats.csv
/home/user/ResCom/Feature_Selection/DataSet/outlier_check/outlier_cases.csv
/home/user/ResCom/Feature_Selection/DataSet/outlier_check/outlier_sample_rows.csv
```

查看最新教师 checkpoint：

```bash
find /home/user/ResCom/results/0708/teacher_full -name best_model.pt | sort | tail -1
```

查看最新学生 checkpoint：

```bash
find /home/user/ResCom/results/0708/student_full -name best_model.pt | sort | tail -1
```

查看 TensorBoard：

```bash
tensorboard --logdir /home/user/ResCom/results/0708
```
