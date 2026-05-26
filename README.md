# Graph Temporal Compensation Project

目录结构：

- train/train_graph_model.py
- infer/infer_graph_model.py
- models/graph_temporal_compensation.py
- models/data_utils.py
- Feature_Selection/DataSet/
  - base_feature_columns.csv
  - merged_error_dataset.csv
  - proxy_feature_columns.csv
  - res_columns.csv
  - target_columns.csv
- Zhurong_Config.json

## 训练

```bash
cd project
python train/train_graph_model.py \
  --feature_dir ./Feature_Selection/DataSet \
  --merged_csv ./Feature_Selection/DataSet/merged_error_dataset.csv \
  --save_dir ./results/graph_temporal \
  --seq_len 20 \
  --batch_size 64 \
  --epochs 80
```

## 推理

```bash
cd project
python infer/infer_graph_model.py \
  --feature_dir ./Feature_Selection/DataSet \
  --merged_csv ./Feature_Selection/DataSet/merged_error_dataset.csv \
  --ckpt ./results/graph_temporal/best_model.pt \
  --out_csv ./infer/predictions.csv \
  --split test \
  --seq_len 20
```

## 模型说明

- 输入按系统、车身、车轮运动、车轮接触四类分组。
- 图结构节点为 1 个系统节点、1 个车身节点、6 个车轮节点。
- 边包括 S-B、B-W_i、左右轮、前中后同侧轮。
- 共享时序主干为 TCN + BiLSTM。
- 输出为车身、车轮运动、接触力学三头残差预测。
- 训练中包含双重监督、物理一致性约束、四元数约束及时序平滑约束。

## 注意

当前代码按列名规则自动分组：
- 车轮前缀默认识别 lf/rf/lm/rm/lb/rb
- 接触类特征默认识别 force/torque/fx/fy/fz/mx/my/mz/slip/sink/contact/load 等关键字
- 若你的真实列名与这些规则差异较大，需要在 `models/data_utils.py` 中调整分组规则
