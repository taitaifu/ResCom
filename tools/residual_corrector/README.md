# Residual Corrector

Two-stage force residual analysis and correction for a trained V4 best model.

## Stage 1: prove residual structure

```bash
python tools/residual_corrector/prove_residual.py \
  --checkpoint results_v4/teacher/20260914_214701/best_output.pt
```

Optional output path:

```bash
python tools/residual_corrector/prove_residual.py \
  --checkpoint results_v4/teacher/20260914_214701/best_output.pt \
  --output_dir tools/residual_corrector/results/prove_name
```

If `--output_dir` is omitted, the script creates `tools/residual_corrector/results/prove_<timestamp>/`.

The saved residual is:

```text
R_true = F_base - F_HF
```

`F_base` is the normal V4 final wheel force inference output, reconstructed through:

```text
build_v4_contact_prediction_raw(): F_final = F_LF + gate_force * pred_force_delta
```

It is not raw delta, `Fphy`, or any other intermediate quantity.

Outputs include:

- `residual_train.npz`
- `residual_val.npz`
- `residual_test.npz`
- `residual_summary.csv`
- `acf_summary.csv`
- `ljung_box_summary.csv`
- `psd_summary.csv`
- `residual_statistics.csv`
- `ar_predictability.csv`
- `counterfactual_summary.csv`
- `proof_summary.json`
- `proof_report.txt`
- `figures/`

## Stage 2: train residual corrector

```bash
python tools/residual_corrector/train_residual_corrector.py \
  --proof_dir tools/residual_corrector/results/prove_name \
  --history_len 50 \
  --residual_history_len 10
```

Default output:

```text
<proof_dir>/corrector_ar/
```

The formal deployable model uses only:

- `F_base_x/y/z`
- `F_LF_x/y/z`
- `Fphy_x/y/z`
- `in_contact`
- `sinkage`
- `slip_long`
- `slip_lat`
- wheel `vel_x/y/z`
- wheel `acc_x/y/z`
- wheel `rel_vel_x/y/z`
- body `acc_x/y/z`

It also uses online predicted residual history:

```text
R_hat(t-10:t-1)
```

It does not use `F_HF`, current `R_true`, future `R_true`, future HF, or any `hf_*` state.

Training uses three stages by default:

- epoch 1-10: teacher forcing, true residual history
- epoch 11-30: scheduled sampling, teacher ratio decays from 1.0 to 0.2
- epoch 31-40: autoregressive fine tune, predicted residual history only

## Oracle diagnostic

```bash
python tools/residual_corrector/train_residual_corrector.py \
  --proof_dir tools/residual_corrector/results/prove_name \
  --oracle_residual_history \
  --history_len 50 \
  --residual_history_len 10
```

Oracle outputs are written to:

```text
<proof_dir>/corrector/oracle/
```

They are marked `ORACLE ONLY - NOT DEPLOYABLE`.

## Online post-processing inference

```bash
python tools/residual_corrector/infer_residual_corrector.py \
  --proof_dir tools/residual_corrector/results/prove_name \
  --corrector_dir tools/residual_corrector/results/prove_name/corrector_ar \
  --split test
```

Online mode initializes the residual buffer with zeros and updates it only with previous `R_hat`.
It writes `F_base`, `R_hat`, and `F_corrected`; it does not read or save HF targets.
