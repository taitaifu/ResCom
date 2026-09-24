# HF Terramechanics Flat Validation

This note documents `validate_hf_terramechanics_flat.py`.

The script is a standalone diagnostic tool. It does not modify the V3 model,
training code, inference code, checkpoint selection, or dataset preprocessing.

It uses HF state data to evaluate the current differentiable terramechanics
formula on a flat terrain assumption:

```text
contact_point  = (0, 0, 0)
contact_normal = (0, 0, 1)
```

The computed force is compared against the HF force columns:

```text
hf_wheel*_Fx
hf_wheel*_Fy
hf_wheel*_Fz
```

## What It Uses

For each wheel, the script reads:

```text
hf body velocity
hf_wheel*_pos_x/y/z
hf_wheel*_omega or hf_wheel*_ang_vel_*
hf_wheel*_Fx/Fy/Fz
```

Then it calls the same implementation used by V3:

```python
models.differentiable_terramechanics.wheel_terrain_force()
```

This keeps the validation aligned with the training/inference physics path.

## Velocity Source

By default, the script uses body velocity as the velocity input to the
terramechanics formula:

```bash
--velocity_source body
```

This maps to:

```text
hf_vel_x/y/z
```

To test whether tangential force errors come from using body velocity instead of
wheel center velocity, run:

```bash
--velocity_source wheel
```

This maps each wheel to:

```text
hf_wheel*_vel_x/y/z
```

This is only a validation/fitting option inside this diagnostic script. It does
not change V3 training or inference.

For SPH HF output, the recommended diagnostic mode is:

```bash
--sph_aligned
```

This applies the definitions documented in
`/home/user/Chrono/RoverSim/SPH_HF_Output_Definitions.md`:

```text
wheel_radius = 0.12
wheel_width  = 0.18
velocity     = hf_wheel*_vel_x/y/z rotated into wheel local frame
omega        = hf_wheel*_ang_vel_x/y/z rotated into wheel local frame, then local Y
force        = formula output compared directly
```

The script also exposes `--force_frame world_from_wheel`, but this is not used
by `--sph_aligned`. A full wheel-quaternion rotation is generally wrong for a
spinning wheel because the C++ terramechanics force basis is a contact/chassis
frame, not the wheel body's continuously spinning frame.

Equivalent explicit options:

```bash
--wheel_radius 0.12 \
--wheel_width 0.18 \
--velocity_source wheel_local \
--angular_source local_y \
--force_frame formula
```

## Pure Validation

Run this to validate the existing default parameters:

```bash
python tools/validate_hf_terramechanics_flat.py \
  --csv Feature_Selection/DataSet/merged_error_dataset.csv \
  --output_dir results_v3/diagnostics/hf_terramechanics_flat_verify \
  --chunksize 200000
```

Compare body velocity and wheel velocity:

```bash
python tools/validate_hf_terramechanics_flat.py \
  --velocity_source body \
  --csv Feature_Selection/DataSet/merged_error_dataset.csv \
  --output_dir results_v3/diagnostics/hf_terramechanics_flat_verify_body \
  --chunksize 200000

python tools/validate_hf_terramechanics_flat.py \
  --velocity_source wheel \
  --csv Feature_Selection/DataSet/merged_error_dataset.csv \
  --output_dir results_v3/diagnostics/hf_terramechanics_flat_verify_wheel \
  --chunksize 200000
```

SPH-aligned validation:

```bash
python tools/validate_hf_terramechanics_flat.py \
  --sph_aligned \
  --tangent_transform_search \
  --csv Feature_Selection/DataSet/merged_error_dataset.csv \
  --output_dir results_v3/diagnostics/hf_terramechanics_flat_verify_sph_aligned \
  --chunksize 200000
```

This mode does not fit or change parameters. It only computes:

```text
flat_phy_force = terramechanics(HF state, flat terrain, current parameters)
```

and compares it with HF force.

## Parameter Fitting

Add `--fit_params` to identify a single global set of flat-terrain parameters:

```bash
python tools/validate_hf_terramechanics_flat.py \
  --fit_params \
  --csv Feature_Selection/DataSet/merged_error_dataset.csv \
  --output_dir results_v3/diagnostics/hf_terramechanics_flat_fit \
  --fit_max_rows 50000 \
  --fit_steps 800 \
  --fit_batch_size 8192 \
  --fit_lr 0.03 \
  --chunksize 200000
```

The fitted parameters are:

```text
Kc, Kphi, n0, n1, c, phi, K
```

The fitting result is written to:

```text
fitted_terrain_params.json
fit_history.json
```

After fitting, the script automatically reruns validation using the fitted
parameters and writes the normal validation metrics.

## Classic Bekker Fitting

The script also supports a separate traditional Bekker/Janosi validation path.
This does not call or modify the V3 model. It fits:

```text
Kc, Kphi, n, c, phi, K
```

Pure classic Bekker fitting:

```bash
python tools/validate_hf_terramechanics_flat.py \
  --sph_aligned \
  --fit_bekker_classic \
  --force_model bekker_classic \
  --csv Feature_Selection/DataSet/merged_error_dataset.csv \
  --output_dir results_v3/diagnostics/hf_bekker_classic_fit \
  --fit_max_rows 100000 \
  --fit_steps 1200 \
  --fit_batch_size 8192 \
  --fit_lr 0.03 \
  --chunksize 200000
```

The result is written to:

```text
fitted_bekker_params.json
bekker_fit_history.json
```

Two optional extensions are kept for comparison with the SPH force behavior:

```bash
--fit_z_offset
--fit_static_fz
--fit_empirical_xy
```

`--fit_static_fz` adds one vertical static-load baseline per wheel:

```text
Fz = static_fz[wheel] + Bekker_Fz
```

`--fit_empirical_xy` keeps Bekker as a dynamic component but adds simple
velocity/slip terms for tangential force:

```text
Fx = a_x * Bekker_Fx + b_x * vx + c_x * slip_long + bias_x[wheel]
Fy = a_y * Bekker_Fy + b_y * vy + c_y * slip_lat  + bias_y[wheel]
```

For the retained hybrid option:

```bash
python tools/validate_hf_terramechanics_flat.py \
  --sph_aligned \
  --fit_bekker_classic \
  --fit_static_fz \
  --fit_empirical_xy \
  --force_model bekker_static_empirical \
  --csv Feature_Selection/DataSet/merged_error_dataset.csv \
  --output_dir results_v3/diagnostics/hf_bekker_static_empirical_fit \
  --fit_max_rows 100000 \
  --fit_steps 1200 \
  --fit_batch_size 8192 \
  --fit_lr 0.03 \
  --chunksize 200000
```

Use the pure `bekker_classic` result first. The static/empirical option is for
checking whether HF is closer to:

```text
Fz = per-wheel static load + Bekker dynamic correction
Fx/Fy = empirical velocity/slip damping
```

## Mean And Gated Baselines

To make the low-order baselines explicit, the script also supports:

```text
wheel_mean
gated_bekker_load_transfer
```

`wheel_mean` fits one constant force vector per wheel:

```bash
python tools/validate_hf_terramechanics_flat.py \
  --sph_aligned \
  --fit_mean_force \
  --force_model wheel_mean \
  --csv Feature_Selection/DataSet/merged_error_dataset.csv \
  --output_dir results_v3/diagnostics/hf_wheel_mean_force_fit \
  --fit_max_rows 100000 \
  --chunksize 200000
```

`gated_bekker_load_transfer` fits:

```text
F = per-wheel force bias
  + alpha_bekker * Bekker_force
  + body roll/pitch/acc load-transfer terms for Fz
  + simple velocity/omega terms for Fx/Fy
```

`alpha_bekker` is constrained to a small range so Bekker can be suppressed if
it hurts validation.

```bash
python tools/validate_hf_terramechanics_flat.py \
  --sph_aligned \
  --fit_gated_bekker \
  --force_model gated_bekker_load_transfer \
  --csv Feature_Selection/DataSet/merged_error_dataset.csv \
  --output_dir results_v3/diagnostics/hf_gated_bekker_load_transfer_fit \
  --fit_max_rows 100000 \
  --fit_steps 1200 \
  --fit_batch_size 8192 \
  --fit_lr 0.03 \
  --chunksize 200000
```

The key output file is:

```text
fitted_gated_bekker_load_transfer.json
```

Check `alpha_bekker`: values near zero mean the fitted validation model does
not need Bekker dynamics.

For ablation, toggle the gated components:

```text
--gated_use_bekker 0/1
--gated_use_load_transfer 0/1
--gated_use_xy_empirical 0/1
```

Load-transfer only:

```bash
python tools/validate_hf_terramechanics_flat.py \
  --sph_aligned \
  --fit_gated_bekker \
  --force_model gated_bekker_load_transfer \
  --gated_use_bekker 0 \
  --gated_use_load_transfer 1 \
  --gated_use_xy_empirical 0 \
  --csv Feature_Selection/DataSet/merged_error_dataset.csv \
  --output_dir results_v3/diagnostics/hf_fz_load_transfer_only \
  --fit_max_rows 100000 \
  --fit_steps 1200 \
  --fit_batch_size 8192 \
  --fit_lr 0.03 \
  --chunksize 200000
```

Load-transfer plus Bekker:

```bash
python tools/validate_hf_terramechanics_flat.py \
  --sph_aligned \
  --fit_gated_bekker \
  --force_model gated_bekker_load_transfer \
  --gated_use_bekker 1 \
  --gated_use_load_transfer 1 \
  --gated_use_xy_empirical 0 \
  --csv Feature_Selection/DataSet/merged_error_dataset.csv \
  --output_dir results_v3/diagnostics/hf_fz_load_transfer_plus_bekker \
  --fit_max_rows 100000 \
  --fit_steps 1200 \
  --fit_batch_size 8192 \
  --fit_lr 0.03 \
  --chunksize 200000
```

Load-transfer plus Fx/Fy empirical terms:

```bash
python tools/validate_hf_terramechanics_flat.py \
  --sph_aligned \
  --fit_gated_bekker \
  --force_model gated_bekker_load_transfer \
  --gated_use_bekker 0 \
  --gated_use_load_transfer 1 \
  --gated_use_xy_empirical 1 \
  --csv Feature_Selection/DataSet/merged_error_dataset.csv \
  --output_dir results_v3/diagnostics/hf_fz_load_transfer_plus_xy \
  --fit_max_rows 100000 \
  --fit_steps 1200 \
  --fit_batch_size 8192 \
  --fit_lr 0.03 \
  --chunksize 200000
```

## Case-Level Train/Validation Split

For generalization checks, split by `case_name` instead of random rows:

```text
--case_split_ratio 0.8
--fit_case_split train
--eval_case_split val
```

This fits parameters on train cases and reports metrics on held-out validation
cases. The split is deterministic under `--seed` and is saved to:

```text
case_split.json
```

Recommended load-transfer-only validation:

```bash
python tools/validate_hf_terramechanics_flat.py \
  --sph_aligned \
  --fit_gated_bekker \
  --force_model gated_bekker_load_transfer \
  --gated_use_bekker 0 \
  --gated_use_load_transfer 1 \
  --gated_use_xy_empirical 0 \
  --case_split_ratio 0.8 \
  --fit_case_split train \
  --eval_case_split val \
  --csv Feature_Selection/DataSet/merged_error_dataset.csv \
  --output_dir results_v3/diagnostics/hf_fz_load_transfer_case_val \
  --fit_max_rows 100000 \
  --fit_steps 1200 \
  --fit_batch_size 8192 \
  --fit_lr 0.03 \
  --chunksize 200000
```

## Tangential Force Transform Search

To check whether `Fx/Fy` errors may come from a sign or coordinate convention
mismatch, add:

```bash
--tangent_transform_search
```

The script will evaluate these candidates without changing the main metrics:

```text
identity
neg_fx
neg_fy
neg_fx_fy
swap_xy
swap_neg_x
swap_neg_y
swap_neg_xy
```

The ranking is saved to:

```text
tangent_transform_search.json
```

Example:

```bash
python tools/validate_hf_terramechanics_flat.py \
  --velocity_source wheel \
  --tangent_transform_search \
  --csv Feature_Selection/DataSet/merged_error_dataset.csv \
  --output_dir results_v3/diagnostics/hf_terramechanics_flat_verify_wheel_transform \
  --chunksize 200000
```

## Linear Force Map

To test whether the physics force has a simple multiplicative or affine
relationship with HF force, add:

```bash
--fit_linear_force_map
```

The script fits:

```text
HF ~= scale * phy
HF ~= affine_scale * phy + affine_bias
```

The default scopes are:

```text
overall
axis
wheel
wheel_axis
```

The result is saved to:

```text
force_linear_map.json
```

Example:

```bash
python tools/validate_hf_terramechanics_flat.py \
  --sph_aligned \
  --fit_linear_force_map \
  --tangent_transform_search \
  --csv Feature_Selection/DataSet/merged_error_dataset.csv \
  --output_dir results_v3/diagnostics/hf_terramechanics_flat_linear_sph \
  --chunksize 200000
```

Important fields:

```text
scale
scale_rmse
affine_scale
affine_bias
affine_rmse
corr
r2_affine
```

If `affine_rmse` drops but `corr` and `r2_affine` remain low, the mapping is
mostly learning a constant bias rather than a useful force law.

## Contact Filter

The contact filter controls how contact is decided before outputting force.

Default:

```bash
--contact_filter none
```

This uses only the physics formula's flat-terrain sinkage:

```text
sinkage > contact_threshold
```

Optional:

```bash
--contact_filter hf
```

This additionally requires the HF contact flag:

```text
sinkage > contact_threshold
and
hf_wheel*_in_contact > 0.5
```

Use `--contact_filter hf` when you want to check whether large errors come
from contact-state mismatch.

## Output Files

The script writes:

```text
flat_phy_vs_hf_metrics.json
flat_phy_vs_hf_samples.csv
scatter_Fx.png
scatter_Fy.png
scatter_Fz.png
```

When `--fit_params` is enabled, it also writes:

```text
fitted_terrain_params.json
fit_history.json
```

## Key Metrics

In `flat_phy_vs_hf_metrics.json`:

```text
force_vector_rmse
zero_force_vector_rmse
force_vector_improve_ratio_vs_zero
mean_true_norm
mean_pred_norm
global_cosine
axis.Fx/Fy/Fz.rmse
axis.Fx/Fy/Fz.bias_pred_minus_hf
axis.Fx/Fy/Fz.corr
sinkage_mean
sinkage_max
contact_ratio
```

Interpretation:

```text
force_vector_rmse
```

RMSE between computed physics force and HF force.

```text
zero_force_vector_rmse
```

RMSE of using zero force as prediction. If physics is useful, force RMSE should
be lower than this.

```text
force_vector_improve_ratio_vs_zero
```

`force_vector_rmse / zero_force_vector_rmse`. Lower is better. Values below 1
mean the formula beats zero force.

```text
mean_pred_norm vs mean_true_norm
```

Checks force magnitude. If `mean_pred_norm` is much smaller than
`mean_true_norm`, the physics formula is under-producing force.

```text
axis.*.bias_pred_minus_hf
```

Mean signed bias. For example, a large negative `Fz` bias means computed normal
force is too small.

```text
axis.*.corr
```

Correlation between computed physics force and HF force for that axis. Low
correlation means the formula is not tracking the HF variation, even if the
average magnitude looks acceptable.

## Important Caveat

This script assumes the terrain is a flat plane at `z=0`.

If HF wheel positions are not expressed relative to that same terrain height
reference, the computed sinkage will be wrong. In that case the force mismatch
is a coordinate/reference problem, not necessarily a terramechanics formula
problem.
