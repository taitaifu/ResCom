"""V6 inference: causal LF input only, shared train/infer reconstruction."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from Feature_Selection.datasetProcess import add_prefix, add_rotvec_residuals, read_case_data
from models.data_utils_v6 import (GraphTemporalSequenceDatasetV6,ROCKER_NAMES,WHEEL_IDS,normalize_optional_columns,
    load_column_spec,load_merged_dataset,split_train_val_test_by_case,ensure_time_feature_columns,check_columns_exist)
from models.checkpoint_v6 import read_checkpoint,restore_model,verify_features
from models.runtime_v6 import forward_sequence,state_values,detach_tree,initial_previous
from models.physics_modules_v6 import matrix_quaternion,local_to_world
from models.loss_utils_v6 import PhysicalMetrics
CONTROL_PARAM_ORDER = ['idle_time','accel_time','accel_speed','const_time','sinusoid_time','sinusoid_speed',
    'sinusoid_freq','step_time','dec_time','idle_end','step_low_speed','step_high_speed']
def map_case_basename(case_dir: Path) -> str:
    name = case_dir.name
    return "case" + name.split("_case", 1)[1] if "_case" in name else name


def build_control_params(values) -> Dict[str, float]:
    if not values:
        return {}
    if len(values) != len(CONTROL_PARAM_ORDER):
        raise ValueError(f"--control_params requires {len(CONTROL_PARAM_ORDER)} numbers")
    return {k: float(v) for k, v in zip(CONTROL_PARAM_ORDER, values)}


def build_custom_case(args, spec) -> pd.DataFrame:
    if not args.custom_case_dir:
        raise ValueError("--custom_case_dir is required for custom_case mode")
    case_name = args.infer_case_name or map_case_basename(Path(args.custom_case_dir))
    controls = build_control_params(args.control_params)
    lf = read_case_data(args.custom_case_dir, controls, compute_features=True)
    if lf is None or lf.empty:
        raise ValueError(f"failed to load Custom case: {args.custom_case_dir}")
    lf = add_prefix(lf.copy(), "lf_")
    lf["case_name"] = case_name
    if args.sph_case_dir:
        hf = read_case_data(args.sph_case_dir, controls, compute_features=False)
        if hf is None or hf.empty:
            raise ValueError(f"failed to load HF reference: {args.sph_case_dir}")
        hf = add_prefix(hf.copy(), "hf_")
        hf["case_name"] = case_name
        start = max(float(lf.time.min()), float(hf.time.min()))
        end = min(float(lf.time.max()), float(hf.time.max()))
        if end < start:
            raise ValueError("Custom and HF reference time ranges do not overlap")
        lf = lf[(lf.time >= start) & (lf.time <= end)].sort_values("time")
        hf = hf[(hf.time >= start) & (hf.time <= end)].sort_values("time")
        df = pd.merge_asof(lf, hf, on="time", by="case_name", direction="nearest")
        residuals = {}
        for hf_col in spec.target_cols:
            if hf_col.startswith("hf_"):
                lf_col = "lf_" + hf_col[3:]
                if lf_col in df and hf_col in df:
                    residuals["res_" + hf_col[3:]] = pd.to_numeric(df[hf_col], errors="coerce").fillna(0) - pd.to_numeric(df[lf_col], errors="coerce").fillna(0)
        if residuals:
            df = pd.concat([df, pd.DataFrame(residuals, index=df.index)], axis=1)
        df = add_rotvec_residuals(df, body=True, wheels=range(6))
    else:
        df = lf
    df = normalize_optional_columns(df, spec, require_labels=False)
    check_columns_exist(df,spec.base_feature_cols,'LF base features')
    check_columns_exist(df,spec.proxy_feature_cols,'LF proxy features')
    return ensure_time_feature_columns(df, "case_name", "time").sort_values(["case_name", "time"]).reset_index(drop=True)


def discover_case_dirs(parent: str, token: str) -> Dict[str, Path]:
    root = Path(parent)
    if not root.is_dir():
        raise FileNotFoundError(f"case directory not found: {root}")
    result = {}
    for child in sorted(root.iterdir()):
        if child.is_dir() and token in child.name and "_case" in child.name:
            name = map_case_basename(child)
            if name in result:
                raise ValueError(f"duplicate case label {name}: {result[name]} and {child}")
            result[name] = child
    if not result:
        raise ValueError(f"no {token} case directories in {root}")
    return result


def build_custom_cases(args, spec) -> pd.DataFrame:
    custom = discover_case_dirs(args.custom_cases_dir, "Custom")
    sph = discover_case_dirs(args.sph_cases_dir, "SPH") if args.sph_cases_dir else {}
    missing = sorted(set(custom) - set(sph)) if sph else []
    if missing:
        raise ValueError(f"Custom cases missing matching HF references: {missing[:10]}")
    frames = []
    for case_name, case_dir in tqdm(custom.items(), desc="Build V6 custom cases", dynamic_ncols=True):
        case_args = argparse.Namespace(**vars(args))
        case_args.custom_case_dir = str(case_dir)
        case_args.sph_case_dir = str(sph[case_name]) if sph else None
        case_args.infer_case_name = case_name
        frames.append(build_custom_case(case_args, spec))
    return pd.concat(frames, ignore_index=True).sort_values(["case_name", "time"]).reset_index(drop=True)




def causal_sample(dataset,t,cfg,device):
    """No forward label window: every slice terminates at current t, including tails."""
    if not hasattr(dataset,'v6_raw'): dataset._prepare_v6_cache()
    h=cfg.history_len;df=dataset.df
    dt=float(df.iloc[t].time-df.iloc[t-1].time)
    if dt<=0 or df.iloc[t].case_name!=df.iloc[t-1].case_name: raise ValueError('invalid causal case boundary/dt')
    batch={}
    for name,arr in dataset.v6_raw.items():
        batch['v6_'+name]=torch.as_tensor(arr[t:t+1],device=device).unsqueeze(0)
        batch['v6_previous_'+name]=torch.as_tensor(arr[t-1:t],device=device).unsqueeze(0)
    keys=['system','body',*ROCKER_NAMES,*[f'wheel{i}_{s}' for i in WHEEL_IDS for s in ('kin','contact')]]
    for role in ('student','teacher'):
        for key in keys:
            batch[f'v6_inputs_{role}_{key}']=torch.as_tensor(dataset.group_arrays[key][t-h:t+1],device=device).unsqueeze(0).unsqueeze(0)
        batch[f'v6_inputs_{role}_physics_nodes']=torch.tensor(dataset.v6_physical[t-h:t+1].copy(),device=device).unsqueeze(0).unsqueeze(0)
    for i in WHEEL_IDS:
        for key,arr in ((f'contact_raw_wheel{i}',dataset.v6_contact[i]),(f'shear_history_wheel{i}',dataset.v6_shear[i])):
            v=arr[t:t+1] if key.startswith('contact_raw') else arr[t-h:t+1]
            batch['v6_inputs_'+key]=torch.tensor(v.copy(),device=device).unsqueeze(0).unsqueeze(0)
        batch[f'v6_inputs_force_history_wheel{i}']=torch.as_tensor(dataset._contact_history_scaled(i,t),device=device).unsqueeze(0).unsqueeze(0)
    batch['v6_dt']=torch.tensor([[dt]],device=device)
    # Training/inference both have assembly anchors derived solely from first LF pose.
    for k,v in dataset.assembly_meta.get(str(df.iloc[t].case_name),{}).items():
        batch['assembly_'+k]=torch.as_tensor(v,device=device).unsqueeze(0)
    return batch


def write_vector(row,name,value,axes=('x','y','z')):
    for key,v in zip(axes,value.detach().cpu().reshape(-1).tolist()): row[name+'::'+key]=v


def run_inference(args):
    ckpt=read_checkpoint(args.checkpoint)
    model,spec,scaler,cfg=restore_model(ckpt,args.device)
    verify_features(ckpt,load_column_spec(args.feature_dir))
    if args.history_len is not None and args.history_len!=cfg.history_len: raise ValueError('history_len differs from checkpoint')
    if args.seed is not None and args.seed!=ckpt['args']['seed']: raise ValueError('split seed differs from checkpoint')
    if args.mode!='auto':
        expected='state_teacher' if model.role=='teacher' else 'force' if ckpt['load_transfer_params'] is not None else 'state_student'
        if args.mode!=expected: raise ValueError('inference mode differs from checkpoint role/stage')
    if args.infer_mode=='merged_csv':
        df=load_merged_dataset(args.merged_csv,spec,require_labels=False,config=cfg)
        if args.split!='all':
            manifest=ckpt['split_manifest']
            ids=manifest[args.split]
            available=set(df.case_name.astype(str).unique())
            if not set(ids)<=available: raise ValueError('checkpoint split cases missing in merged_csv')
            df=df[df.case_name.astype(str).isin(ids)].copy()
    elif args.infer_mode=='custom_case': df=build_custom_case(args,spec)
    else: df=build_custom_cases(args,spec)
    if args.case_name: df=df[df.case_name.astype(str)==str(args.case_name)].copy()
    if df.empty: raise ValueError('no cases selected')
    df=normalize_optional_columns(df,spec,require_labels=False,spin_axis=cfg.spin_axis,wheel_radius=cfg.wheel_radius)
    dataset=GraphTemporalSequenceDatasetV6(df,spec,scaler,history_len=cfg.history_len,teacher_future_len=0,body_rollout_len=1)
    dataset.config=cfg;model.eval();params=ckpt['load_transfer_params']
    records=[];metrics=PhysicalMetrics();previous=None;last_case=None;last_inputs=None
    positions=[];horizon=0;accumulation=[]
    targets=[]
    for _,ix in dataset.df.groupby('case_name',sort=False).indices.items(): targets.extend(np.asarray(ix)[cfg.history_len:].tolist())
    if args.max_samples is not None: targets=targets[:args.max_samples]
    for t in tqdm(targets,desc=f'V6 {args.rollout_mode} {args.split}',dynamic_ncols=True):
        rowmeta=dataset.df.iloc[t];case=str(rowmeta.case_name)
        if case!=last_case: previous=None;last_inputs=None;horizon=0
        batch=causal_sample(dataset,t,cfg,args.device)
        # Autonomous continuous inference replaces motion observations by the
        # previous predicted state and propagates model-fed histories.
        if args.rollout_mode=='autonomous' and previous is not None:
            from models.runtime_v6 import make_step_inputs,autonomous_inputs,_part_names
            inputs=make_step_inputs(batch,0,'student')
            raw={part:batch[f'v6_lf_{part}'][:,0:1] for part in _part_names()}
            dt=batch['v6_dt'][...,None]
            inputs,raw=autonomous_inputs(inputs,raw,previous,dt,spec,scaler,cfg)
            for key in ['system','body',*ROCKER_NAMES,*[f'wheel{i}_{s}' for i in WHEEL_IDS for s in ('kin','contact')],'physics_nodes']:
                inputs[key]=torch.cat([last_inputs[key][:,1:],inputs[key][:,-1:]],1)
                for role in ('student','teacher'): batch[f'v6_inputs_{role}_{key}']=inputs[key].unsqueeze(1)
            for part in _part_names(): batch[f'v6_lf_{part}']=raw[part]
        with torch.no_grad():
            states,forces=forward_sequence(model,batch,spec,scaler,cfg,params,1,model.role,previous=previous)
        state=states[0];force=forces[0];previous=detach_tree(state);last_case=case
        from models.runtime_v6 import make_step_inputs
        last_inputs=make_step_inputs(batch,0,'student');horizon+=1
        row={'case_name':case,'time':float(rowmeta.time),'step':horizon,'rollout_mode':args.rollout_mode}
        for part in ('body',*[f'rocker_{n}' for n in ROCKER_NAMES],*[f'wheel{i}' for i in WHEEL_IDS]):
            p,v,r,w,o=state_values(state,part)
            lf=torch.as_tensor(dataset.v6_raw['lf_'+part][t:t+1],device=args.device).unsqueeze(0)
            hf=torch.as_tensor(dataset.v6_raw['hf_'+part][t:t+1],device=args.device).unsqueeze(0)
            mask=torch.as_tensor(dataset.v6_raw['hf_'+part+'_mask'][t:t+1],device=args.device).unsqueeze(0)
            from models.physics_modules_v6 import rotation_error,quat_matrix
            for kind,pred,j,scale,tol in [('position',p,0,cfg.position_scale,1e-5),('velocity',v,3,cfg.velocity_scale,1e-5),('angular_velocity',w,10,cfg.angular_velocity_scale,1e-5)]:
                write_vector(row,f'pred::{part}::{kind}',pred);write_vector(row,f'lf::{part}::{kind}',lf[...,j:j+3])
                if bool(mask[...,j:j+3].all()): write_vector(row,f'true::{part}::{kind}',hf[...,j:j+3])
                metrics.add(part+'/'+kind,pred,hf[...,j:j+3],lf[...,j:j+3],mask[...,j:j+3],tol,scale)
                metrics.add(f'rollout/{args.rollout_mode}/step_{horizon}/{part}/{kind}',pred,hf[...,j:j+3],lf[...,j:j+3],mask[...,j:j+3],tol,scale)
            write_vector(row,f'pred::{part}::quaternion',matrix_quaternion(r),('q0','q1','q2','q3'))
            write_vector(row,f'lf::{part}::quaternion',lf[...,6:10],('q0','q1','q2','q3'))
            err=rotation_error(r,quat_matrix(hf[...,6:10]));baseerr=rotation_error(quat_matrix(lf[...,6:10]),quat_matrix(hf[...,6:10]))
            metrics.add(part+'/attitude_rad',err,torch.zeros_like(err),baseerr,mask[...,6:10].amin(-1,keepdim=True),1e-5,.1)
            if o is not None:
                write_vector(row,f'pred::{part}::omega',o,('omega',));write_vector(row,f'lf::{part}::omega',lf[...,13:14],('omega',))
                metrics.add(part+'/omega',o,hf[...,13:14],lf[...,13:14],mask[...,13:14],1e-4,cfg.omega_scale)
                row[f'lf::{part}::omega_valid']=bool(dataset.v6_raw['lf_'+part+'_mask'][t,13])
                row[f'true::{part}::omega_valid']=bool(mask[...,13:14].all())
            out=state['state_output'];gatekey='body' if part=='body' else part
            gate=out[f'gate_{gatekey}_pos']
            raw=state['raw_body_position'] if part=='body' else out[f'pred_{part}_local_delta_pos'] if part.startswith('rocker_') else state[f'raw_{part}_local']
            metrics.gate(part+'/position',raw,gate)
            write_vector(row,f'raw_residual::{part}::position',raw);write_vector(row,f'gate::{part}::position',gate)
            write_vector(row,f'effective_residual::{part}::position',raw*gate)
            att_raw=out['pred_delta_rot_body'] if part=='body' else out[f'pred_{part}_delta_rot']
            att_gate=out[f'gate_{gatekey}_att']
            metrics.gate(part+'/attitude',att_raw,att_gate)
            write_vector(row,f'gate::{part}::attitude',att_gate)
            write_vector(row,f'raw_residual::{part}::attitude',att_raw)
            write_vector(row,f'effective_residual::{part}::attitude',att_raw*att_gate)
            if part=='body':
                for kind,raw,g in [('velocity',state['raw_body_velocity'],out['gate_body_velocity']),
                    ('angular_velocity',out['delta_body_angular_velocity']*cfg.angular_velocity_scale,out['gate_body_angular_velocity'])]:
                    metrics.gate('body/'+kind,raw,g)
                    write_vector(row,'gate::body::'+kind,g)
                    write_vector(row,'raw_residual::body::'+kind,raw)
                    write_vector(row,'effective_residual::body::'+kind,raw*g)
            elif part.startswith('wheel'):
                i=int(part[5:]);og=out[f'gate_wheel{i}_omega'];oraw=state[f'raw_wheel{i}_omega']
                metrics.gate(part+'/omega',oraw,og,o-lf[...,13:14])
                write_vector(row,f'gate::{part}::omega',og,('omega',))
                write_vector(row,f'raw_residual::{part}::omega',oraw,('omega',))
                write_vector(row,f'effective_residual::{part}::omega',oraw*og if model.omega_supervised[i] else torch.zeros_like(oraw),('omega',))
        from models.runtime_v6 import assembly_rollout_loss
        assembly=assembly_rollout_loss(batch,state['body_position'],state['body_rotation'],state['rockers'],state['wheel_position'])
        row['assembly_error']=float(assembly)
        for i in WHEEL_IDS:
            if force is None: continue
            pred=force[f'force_pred_wheel{i}'];lf=state['lf_force'][i]
            truth=batch[f'v6_hf_force_wheel{i}'][:,0:1];mask=batch[f'v6_hf_force_wheel{i}_mask'][:,0:1]
            if cfg.input_force_frame=='world':
                from models.physics_modules_v6 import world_to_local
                truth=world_to_local(truth,force[f'contact_frame_wheel{i}'])
                mask=mask.amin(-1,keepdim=True).expand_as(mask)
            write_vector(row,f'pred_force::wheel{i}',pred,('Fx','Fy','Fz'))
            write_vector(row,f'lf_force::wheel{i}',lf,('Fx','Fy','Fz'))
            write_vector(row,f'hybrid_force::wheel{i}',state['force_base'][i],('Fx','Fy','Fz'))
            write_vector(row,f'pred_force_world::wheel{i}',local_to_world(pred,force[f'contact_frame_wheel{i}']))
            write_vector(row,f'gate_force::wheel{i}',force[f'gate_force_wheel{i}'],('Fx','Fy','Fz'))
            write_vector(row,f'raw_force_residual::wheel{i}',force[f'delta_force_wheel{i}'],('Fx','Fy','Fz'))
            write_vector(row,f'effective_force_residual::wheel{i}',force[f'effective_force_wheel{i}'],('Fx','Fy','Fz'))
            if bool(mask.all()): write_vector(row,f'true_force::wheel{i}',truth,('Fx','Fy','Fz'))
            metrics.gate(f'wheel{i}/force',force[f'delta_force_wheel{i}'],force[f'gate_force_wheel{i}'],force[f'effective_force_wheel{i}'])
            for j,a in enumerate(('Fx','Fy','Fz')):
                metrics.add(f'wheel{i}/force_{a}',pred[...,j:j+1],truth[...,j:j+1],lf[...,j:j+1],mask[...,j:j+1],1e-3,cfg.force_scale)
                metrics.add(f'wheel{i}/baseline_hybrid_{a}',state['force_base'][i][...,j:j+1],truth[...,j:j+1],lf[...,j:j+1],mask[...,j:j+1],1e-3,cfg.force_scale)
        if args.output_intermediates and force is not None:
            for k in ('contact_resultant_world','contact_torque_world','contact_resultant_delta_world','contact_torque_delta_world'):
                if k in force: write_vector(row,k,force[k])
        records.append(row)
    if not records: raise ValueError('no rows after history warm-up')
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(output/'predictions_v6.csv',index=False)
    summary={'checkpoint':str(Path(args.checkpoint).resolve()),'split':args.split,'infer_mode':args.infer_mode,
        'rollout_mode':args.rollout_mode,'uses_future_lf':False,'uses_hf_as_input':False,'config':cfg.to_dict(),
        'num_rows':len(records),'metrics':metrics.summarize(),
        'autonomous_scope':'predicted state recurrence; current LF controls, terrain and contact remain observed',
        'truncated':args.max_samples is not None}
    (output/'metrics_v6.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(f'Predictions: {output / "predictions_v6.csv"}\nReport: {output / "metrics_v6.json"}',flush=True)
    return summary


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--mode',choices=['auto','state_teacher','state_student','force'],default='auto')
    p.add_argument('--feature_dir',default=str(ROOT/'Feature_Selection/DataSet'))
    p.add_argument('--merged_csv',default=str(ROOT/'Feature_Selection/DataSet/merged_error_dataset.csv'))
    p.add_argument('--infer_mode',choices=['merged_csv','custom_case','custom_cases'],default='merged_csv')
    p.add_argument('--split',choices=['all','train','val','test'],default='test')
    for k in ('case_name','custom_case_dir','sph_case_dir','custom_cases_dir','sph_cases_dir','infer_case_name'): p.add_argument('--'+k)
    p.add_argument('--control_params',type=float,nargs=12)
    p.add_argument('--history_len',type=int);p.add_argument('--seed',type=int)
    p.add_argument('--max_samples',type=int)
    p.add_argument('--rollout_mode',choices=['lf_conditioned','autonomous'],default='lf_conditioned')
    p.add_argument('--output_intermediates',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output_dir',default=str(ROOT/'results_v6/inference')+'/'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    args=p.parse_args()
    if args.infer_mode=='custom_case' and not args.custom_case_dir: p.error('--custom_case_dir required')
    if args.infer_mode=='custom_cases' and not args.custom_cases_dir: p.error('--custom_cases_dir required')
    return args


if __name__=='__main__': run_inference(parse_args())
