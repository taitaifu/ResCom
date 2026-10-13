"""Independent V6 staged training/evaluation; no code executes on import."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import math
import random
import sys
from time import perf_counter
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models.config_v6 import V6Config, add_config_arguments, config_from_args
from models.data_utils_v6 import (load_column_spec,load_merged_dataset,split_train_val_test_by_case,
    GroupStandardizer,GraphTemporalSequenceDatasetV6,graph_temporal_collate_fn,save_column_spec_json)
from models.checkpoint_v6 import (build_model,read_checkpoint,verify_features,compatible_load,
    restore_scaler,restore_model,atomic_save,ARCH_KEYS,physics_resource_fingerprints)
from models.load_transfer_force_v6 import fit_load_transfer_params
from models.loss_utils_v6 import LossLedger,PhysicalMetrics
from models.runtime_v6 import forward_sequence,state_losses,force_losses,make_step_inputs

STAGES=('state_teacher','state_student','force','physics','joint')
FORCE_PREFIXES=('force_history_encoder.','force_residual_head.','enhanced_history.','contact_edge_decoder.','force_gate_head.')
DYNAMICS_PREFIXES=('dynamics_head.',)


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    add_config_arguments(p)
    p.add_argument('--train_stage',choices=STAGES,default='state_teacher')
    p.add_argument('--feature_dir',default=str(ROOT/'Feature_Selection/DataSet'))
    p.add_argument('--merged_csv',default=str(ROOT/'Feature_Selection/DataSet/merged_error_dataset.csv'))
    p.add_argument('--save_dir',default=str(ROOT/'results_v6'))
    p.add_argument('--run_dir')
    p.add_argument('--init_ckpt');p.add_argument('--state_checkpoint');p.add_argument('--force_checkpoint');p.add_argument('--resume_ckpt')
    p.add_argument('--eval_checkpoint');p.add_argument('--eval_split',choices=['val','test'],default='val')
    p.add_argument('--report_path')
    p.add_argument('--check_config',action='store_true');p.add_argument('--prepare_only',action='store_true')
    p.add_argument('--history_len',type=int,default=9)
    p.add_argument('--teacher_future_len',type=int,default=3)
    p.add_argument('--body_rollout_len',type=int,default=5)
    p.add_argument('--batch_size',type=int,default=64);p.add_argument('--grad_accum_steps',type=int,default=1)
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--patience',type=int,default=15,help='stop a stage after N consecutive epochs without a lower validation selection_score')
    p.add_argument('--weight_decay',type=float,default=1e-5)
    for key,value in [('hidden_dim',192),('graph_layers',3),('tcn_dim',256),('lstm_dim',192),('lstm_layers',2),('force_hidden_dim',48)]: p.add_argument('--'+key,type=int,default=value)
    p.add_argument('--dropout',type=float,default=.1)
    for key,value in [('w_state',1.),('w_force',1.),('w_physics',1.),('w_gate',.1),('w_rollout',.05),
        ('lambda_improve',.05),('improve_margin',0.),('lambda_assembly',.02),('lambda_kin',.005),('lambda_dyn',0.),
        ('lambda_wheel_local_reg',1e-4),('lambda_body_delta_reg',1e-3),('lambda_body_delta_smooth',.01),
        ('lambda_force_residual_reg',1e-5),('lambda_distill_z',1.)]: p.add_argument('--'+key,type=float,default=value)
    p.add_argument('--state_trainable_layers',nargs='+',default=['body_velocity_head','body_position_correction_head','body_att_head','body_angular_velocity_head','wheel_local_position_head','wheel_spin_head','part_velocity_head','part_angular_head','hgt_layers.2'])
    p.add_argument('--joint_freeze_state_heads',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--num_workers',type=int,default=0);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--grad_clip',type=float,default=5.)
    p.add_argument('--gradient_log_interval',type=int,default=100,help='sample per-loss effective gradients every N batches and at epoch end; 1=every batch')
    p.add_argument('--max_batches',type=int,default=None,help='manual CPU/data smoke runs only; reports mark truncation')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=p.parse_args(argv)
    if a.batch_size<1 or a.grad_accum_steps<1 or a.epochs<1: p.error('batch_size/grad_accum_steps/epochs must be positive')
    if a.patience<1: p.error('patience must be positive')
    if not 2<=a.body_rollout_len<=5: p.error('body_rollout_len must be 2..5')
    if a.gradient_log_interval<1: p.error('gradient_log_interval must be positive')
    if a.num_workers<0: p.error('num_workers must be nonnegative')
    if a.max_batches is not None and a.max_batches<1: p.error('max_batches must be positive')
    return a


def move(batch,device):
    return {k:v.to(device,non_blocking=str(device).startswith('cuda')) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}


def make_loader(dataset,args,shuffle=False):
    if dataset is None: return None
    options={}
    if args.num_workers>0:
        import multiprocessing
        if 'fork' not in multiprocessing.get_all_start_methods():
            raise ValueError('V6 in-memory dataset workers require fork sharing; use --num_workers 0 on this platform')
        if hasattr(dataset,'prepare_worker_cache'):
            dataset.prepare_worker_cache()
        # Each batch contains hundreds of independent tensor storages. The
        # default file_descriptor strategy can exhaust the process FD limit.
        torch.multiprocessing.set_sharing_strategy('file_system')
        options.update(prefetch_factor=1,multiprocessing_context='fork')
        print('DataLoader: parent-prepared cache, fork/file_system sharing, prefetch_factor=1; num_workers='+str(args.num_workers),flush=True)
    return DataLoader(dataset,batch_size=args.batch_size,shuffle=shuffle,
        num_workers=args.num_workers,collate_fn=graph_temporal_collate_fn,
        pin_memory=args.device.startswith('cuda'),**options)

def json_write(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def data_bundle(args,cfg,checkpoint=None):
    spec=load_column_spec(args.feature_dir)
    df=load_merged_dataset(args.merged_csv,spec,config=cfg)
    train,val,test=split_train_val_test_by_case(df,seed=args.seed)
    scaler=GroupStandardizer().fit(train,spec) if checkpoint is None else restore_scaler(checkpoint)
    if checkpoint is not None: verify_features(checkpoint,spec)
    sets=[]
    for part in (train,val,test):
        ds=GraphTemporalSequenceDatasetV6(part,spec,scaler,history_len=cfg.history_len,
            teacher_future_len=args.teacher_future_len,body_rollout_len=args.body_rollout_len) if len(part) else None
        if ds is not None:
            ds.config=cfg
            ds.compact_training_samples=True
        sets.append(ds)
    split={k:sorted(d.case_name.astype(str).unique().tolist()) for k,d in zip(('train','val','test'),(train,val,test))}
    if checkpoint is not None and checkpoint.get('split_manifest') and split!=checkpoint['split_manifest']:
        raise ValueError('case split differs from checkpoint')
    return spec,scaler,train,sets,split


def stage_trainable(model,args):
    stage=args.train_stage
    for name,p in model.named_parameters():
        force=name.startswith(FORCE_PREFIXES);dyn=name.startswith(DYNAMICS_PREFIXES)
        if stage=='force': active=force
        elif stage=='physics': active=dyn
        elif stage=='joint':
            selected=any(name.startswith(x) for x in args.state_trainable_layers)
            if args.joint_freeze_state_heads and ('head.' in name or '_head.' in name): selected=False
            active=force or dyn or selected
        else: active=not (force or dyn)
        # Bypass modules must not produce useless optimizer entries.
        if model.config.enable_contact_edge_decoder and name.startswith('force_residual_head.'): active=False
        if model.config.enable_contact_history and name.startswith('force_history_encoder.'): active=False
        if name.startswith('wheel_omega_head.') and not model.uses_legacy_spin_head: active=False
        if name.startswith('wheel_spin_head.') and model.uses_legacy_spin_head: active=False
        p.requires_grad_(active)
    if not any(p.requires_grad for p in model.parameters()): raise ValueError('no trainable parameters for selected stage')


def compute_batch(model,batch,stage,teacher,params,spec,scaler,cfg,args,physical):
    role='teacher' if stage=='state_teacher' else 'student'
    steps=args.body_rollout_len
    # State teacher/student bypass all force/coupling modules entirely.
    use_force=stage in ('force','physics','joint')
    states,forces=forward_sequence(model,batch,spec,scaler,cfg,params if use_force else None,steps,role)
    ledger=LossLedger(states[0]['body_position'])
    # Force stage backbone parameters are frozen; diagnostics remain visible.
    state_losses(states,batch,spec,scaler,cfg,args,ledger,physical)
    if use_force: force_losses(states,forces,batch,cfg,args,ledger,physical)
    if teacher is not None:
        from models.data_utils_v6 import input_only
        with torch.no_grad():
            t=teacher.forward_state(input_only(make_step_inputs(batch,0,'teacher'),'teacher'),current_index=cfg.history_len)
        z=states[0]['state_output']['z']
        if z.shape!=t['z'].shape: raise ValueError('Teacher/Student latent shapes differ')
        ledger.add('distill',torch.nn.functional.smooth_l1_loss(z,t['z']),args.w_state*args.lambda_distill_z)
    else:
        ledger.add('distill',states[0]['body_position'].sum()*0,args.lambda_distill_z,False)
    if cfg.enable_pushforward and stage in ('state_student','physics','joint'):
        pushed,pforce=forward_sequence(model,batch,spec,scaler,cfg,params if use_force else None,
            min(cfg.rollout_steps,args.body_rollout_len),role,autonomous=True)
        state_losses(pushed,batch,spec,scaler,cfg,args,ledger,physical,'pushforward',args.w_rollout)
        if use_force: force_losses(pushed,pforce,batch,cfg,args,ledger,physical,'pushforward_force',args.w_rollout)
    else:
        ledger.add('pushforward',states[0]['body_position'].sum()*0,args.w_rollout,False)
    if not use_force:
        ledger.add('force/dyn',states[0]['body_position'].sum()*0,args.lambda_dyn,False)
        ledger.add('force/gate',states[0]['body_position'].sum()*0,args.w_gate,False)
    return ledger


def run_epoch(model,loader,args,cfg,spec,scaler,params,teacher=None,optimizer=None,epoch=0,stage=None):
    training=optimizer is not None;stage=stage or args.train_stage
    model.train(training)
    # Freeze dropout/statistics along with frozen state modules.
    if training:
        for m in model.modules():
            p=list(m.parameters(recurse=True))
            if p and not any(x.requires_grad for x in p): m.eval()
    if teacher is not None: teacher.eval()
    physical=PhysicalMetrics(defer_transfer=True);sums={};counts={};samples=0;gradient_batches=0
    timing={key:0. for key in ('data_wait','transfer_enqueue','forward_loss_log','gradient_backward_optimizer','metric_transfer')}
    if loader is None or len(loader)==0: raise ValueError('selected split has no valid sequence windows')
    num=len(loader) if args.max_batches is None else min(len(loader),args.max_batches)
    if training: optimizer.zero_grad(set_to_none=True)
    progress=tqdm(total=num,desc=f'{stage} {"train" if training else "val"} Epoch {epoch}',dynamic_ncols=True)
    iterator=iter(loader)
    for index in range(num):
        start=perf_counter();cpu_batch=next(iterator)
        timing['data_wait']+=perf_counter()-start
        start=perf_counter();batch=move(cpu_batch,args.device);size=batch['body'].shape[0]
        timing['transfer_enqueue']+=perf_counter()-start
        start=perf_counter()
        with torch.set_grad_enabled(training):
            ledger=compute_batch(model,batch,stage,teacher,params,spec,scaler,cfg,args,physical)
            scalar=ledger.scalars()
            if not math.isfinite(scalar['total']): raise FloatingPointError('non-finite loss')
            timing['forward_loss_log']+=perf_counter()-start
            start=perf_counter()
            if training:
                if (index+1)%args.gradient_log_interval==0 or index+1==num:
                    scalar.update(ledger.gradients(model.parameters()));gradient_batches+=1
                group_start=(index//args.grad_accum_steps)*args.grad_accum_steps
                group_end=min(group_start+args.grad_accum_steps,num)
                # Exact sample-weighted microbatch accumulation, including final tail.
                end_samples=min(group_end*args.batch_size,len(loader.dataset))
                start_samples=group_start*args.batch_size
                denom=max(end_samples-start_samples,size)
                (ledger.total*(size/denom)).backward()
                if index+1==group_end:
                    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],args.grad_clip)
                    if not torch.isfinite(norm): raise FloatingPointError('non-finite gradient')
                    optimizer.step();optimizer.zero_grad(set_to_none=True)
            timing['gradient_backward_optimizer']+=perf_counter()-start
        start=perf_counter();physical.flush()
        timing['metric_transfer']+=perf_counter()-start
        del cpu_batch
        samples+=size
        for k,v in scalar.items():
            if isinstance(v,(int,float,bool)):
                sums[k]=sums.get(k,0.)+float(v)*size;counts[k]=counts.get(k,0)+size
        progress.update()
        progress.set_postfix(Loss=f"{scalar['total']:.5g}",Body_MAE=f"{physical.quick_mae('body/position'):.4g}",
            Fy_MAE=f"{physical.quick_mae('wheel0/force_Fy'):.4g}",LR=f"{optimizer.param_groups[0]['lr']:.3g}" if training else '-')
    progress.close()
    losses={k:v/counts[k] for k,v in sums.items()}
    for k in list(losses):
        if k.endswith('/enabled'): losses[k]=bool(losses[k]>.5)
    if not training:
        for name,(_,_,enabled) in ledger.entries.items():
            losses[f'loss/{name}/effective_gradient']=None if enabled else 0.
    metrics=physical.summarize()
    physical_score=metrics['selection_score']
    assembly=losses.get('loss/state/assembly/raw',0.)
    if physical_score is None: raise ValueError('no physical-unit validation labels available')
    return {'losses':losses,'metrics':metrics,'selection_score':physical_score+.01*assembly,
        'samples':samples,'truncated':args.max_batches is not None,
        'gradient_status':'sampled' if training else 'not_computed_no_grad',
        'gradient_batches':gradient_batches,'gradient_log_interval':args.gradient_log_interval,
        'timing_host_seconds':timing,'timing_host_seconds_per_batch':{k:v/num for k,v in timing.items()},
        'timing_scope':'host wall time including CUDA waits at existing synchronization points; not kernel profiling'}


def payload(model,spec,scaler,args,cfg,params,split,epoch,best,optimizer=None):
    d={'version':6,'role':model.role,'train_stage':args.train_stage,'epoch':epoch,'best_metric':best,
        'model_state_dict':model.state_dict(),'v6_config':cfg.to_dict(),'architecture':{k:getattr(args,k) for k in ARCH_KEYS},
        'args':vars(args).copy(),'column_spec':spec.to_dict(),'scaler_state':{k:v.to_dict() for k,v in scaler.scalers.items()},
        'load_transfer_params':params,'split_manifest':split,
        'feature_config':{'omega_supervised':model.omega_supervised,'physics_node_dim':32,'contact_dim':15,
            'legacy_contact_history_dim':7,'enhanced_shear_history_dim':8,'input_force_frame':cfg.input_force_frame,
            'terrain_source':cfg.terrain_file,'physics_resource_fingerprints':physics_resource_fingerprints(cfg),
            'causal_force_preprocess':'observed_dt_lowpass_5Hz','student_inputs':'causal_LF_only',
            'physics_baseline_components':['Fz'],'physics_Fx_Fy_status':'no calibrated V5 prior; LF retained'}}
    if optimizer is not None:
        d.update(optimizer_state_dict=optimizer.state_dict(),resume_format='v6_complete_epoch_v1',
            rng_state={'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
                'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None})
    return d


def main(argv=None):
    args=parse_args(argv);cfg=config_from_args(args)
    if args.lambda_dyn>0 and not cfg.dynamics_calibrated: raise ValueError('dyn_loss is disabled until whole-vehicle dynamics is calibrated')
    if args.train_stage=='physics' and not cfg.enable_dynamics_update: raise ValueError('physics stage requires dynamics_update; v6_a skips this stage')
    print(json.dumps({'effective_v6_config':cfg.to_dict()},indent=2),flush=True)
    if args.check_config:
        if args.report_path: json_write(args.report_path,cfg.to_dict())
        return
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if args.eval_checkpoint:
        ckpt=read_checkpoint(args.eval_checkpoint)
        model,spec,scaler,cfg=restore_model(ckpt,args.device)
        saved=ckpt['args']
        for key in ('seed','teacher_future_len','body_rollout_len',*ARCH_KEYS): setattr(args,key,saved[key])
        for key in list(vars(args)):
            if key.startswith(('w_','lambda_')) or key=='improve_margin': setattr(args,key,saved.get(key,getattr(args,key)))
        _,_,_,sets,split=data_bundle(args,cfg,ckpt)
        selected=sets[1 if args.eval_split=='val' else 2]
        loader=make_loader(selected,args)
        report=run_epoch(model,loader,args,cfg,spec,scaler,ckpt.get('load_transfer_params'),stage=ckpt['train_stage'])
        report.update(checkpoint=str(Path(args.eval_checkpoint).resolve()),split=args.eval_split,split_manifest=split,config=cfg.to_dict(),evaluation_protocol='v6_causal_window_physical_units_v1')
        json_write(args.report_path or Path(args.eval_checkpoint).parent/f'{args.eval_split}_report.json',report)
        print(json.dumps(report,ensure_ascii=False),flush=True);return
    source_path=args.resume_ckpt or (args.state_checkpoint if args.train_stage=='force' else args.force_checkpoint if args.train_stage in ('physics','joint') else args.init_ckpt)
    source=read_checkpoint(source_path) if source_path else None
    if args.resume_ckpt:
        if source.get('version')!=6 or source.get('resume_format')!='v6_complete_epoch_v1' or source['train_stage']!=args.train_stage: raise ValueError('invalid stage/resume checkpoint')
        # Resume restores all model/data/loss settings; conflicting structural flags fail.
        saved=source['args'];explicit={t.split('=')[0].replace('--no-','--',1) for t in (argv or sys.argv[1:]) if t.startswith('--')}
        runtime={'epochs','patience','device','num_workers','resume_ckpt','max_batches','gradient_log_interval','report_path'}
        for key,val in saved.items():
            if key in runtime: continue
            if hasattr(args,key) and '--'+key in explicit and getattr(args,key)!=val: raise ValueError(f'resume config override rejected: {key}')
            if hasattr(args,key): setattr(args,key,val)
        cfg=V6Config.from_dict(source['v6_config'])
    if source is not None:
        arch=source.get('architecture',source.get('args',{}))
        explicit={t.split('=')[0] for t in (argv or sys.argv[1:]) if t.startswith('--')}
        for key in ARCH_KEYS:
            if key in arch:
                if '--'+key in explicit and getattr(args,key)!=arch[key]: raise ValueError(f'architecture mismatch {key}')
                setattr(args,key,arch[key])
        if source.get('version')==6 and not args.resume_ckpt and source['v6_config']!=cfg.to_dict():
            # Explicit new ablation stages permit added modules, but presets are not silently restored.
            print('V6 module configuration differs: exact compatible weights transfer; added modules initialize independently',flush=True)
    spec,scaler,train,sets,split=data_bundle(args,cfg,source if args.resume_ckpt else None)
    if source is not None: verify_features(source,spec,scaler)
    if source is not None and source.get('split_manifest') and source['split_manifest']!=split:
        raise ValueError('initialization checkpoint uses a different case split')
    if source is not None and source.get('version')==6:
        source_cfg=V6Config.from_dict(source['v6_config'])
        semantic=('wheel_radius','terrain_file','spin_axis','input_force_frame','omega_scale','velocity_scale',
            'angular_velocity_scale','position_scale','force_scale','torque_scale','shear_rho','mass','gravity','inertia_body',
            'dynamics_calibrated','dynamics_reference_is_vehicle_com')
        for key in semantic:
            if str(getattr(source_cfg,key))!=str(getattr(cfg,key)):
                raise ValueError(f'initialization physics/feature semantics differ: {key}')
        if source['feature_config'].get('physics_resource_fingerprints')!=physics_resource_fingerprints(cfg):
            raise ValueError('initialization geometry/terrain resources changed')
    directory=Path(args.run_dir) if args.run_dir else Path(args.save_dir)/cfg.ablation/args.train_stage/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if args.resume_ckpt: directory=Path(args.resume_ckpt).resolve().parent
    elif directory.exists(): raise FileExistsError(f'run_dir already exists; refuse stale results: {directory}')
    directory.mkdir(parents=True,exist_ok=True)
    json_write(directory/'config_v6.json',cfg.to_dict());json_write(directory/'split_manifest.json',split)
    save_column_spec_json(spec,str(directory/'column_spec_v6.json'));scaler.save(str(directory/'group_scaler_v6.joblib'))
    if args.prepare_only:
        json_write(directory/'data_report.json',{'split_manifest':split,'samples':[len(d) if d else 0 for d in sets],
            'omega_supervised':[bool(train[f'v6_hf_wheel{i}_omega_valid'].any()) for i in range(6)],
            'lf_omega_missing_fraction':[float((~train[f'v6_lf_wheel{i}_omega_valid'].astype(bool)).mean()) for i in range(6)]})
        return
    role='teacher' if args.train_stage=='state_teacher' else 'student'
    model=build_model(spec,args,cfg,role).to(args.device)
    model.omega_supervised=[bool(train[f'v6_hf_wheel{i}_omega_valid'].any()) for i in range(6)]
    teacher=None
    if args.train_stage=='state_student':
        if source is None: raise ValueError('state_student requires --init_ckpt Teacher')
        teacher_source=read_checkpoint(args.init_ckpt) if args.resume_ckpt else source
        if teacher_source.get('train_stage')!='state_teacher': raise ValueError('state_student initialization must be Teacher')
        if teacher_source.get('version')==6:
            teacher,_,_,_=restore_model(teacher_source,args.device)
        else:
            teacher=build_model(spec,args,cfg,'teacher').to(args.device)
            compatible_load(teacher,teacher_source,report_path=directory/'teacher_load.json')
        teacher.eval()
        for p in teacher.parameters(): p.requires_grad_(False)
    if source is not None:
        if args.resume_ckpt: model.load_state_dict(source['model_state_dict'],strict=True)
        else: compatible_load(model,source,teacher_to_student=args.train_stage=='state_student',report_path=directory/'checkpoint_load.json')
    elif args.train_stage in ('force','physics','joint'): raise ValueError('stage requires previous checkpoint')
    params=None
    if args.train_stage in ('force','physics','joint'):
        params=source.get('load_transfer_params') if source else None
        if params is None:
            if args.train_stage!='force': raise ValueError('force baseline missing; run force stage first')
            params=fit_load_transfer_params(train,config=cfg)
    stage_trainable(model,args)
    groups=[]
    for kind,prefixes,factor in [('force',FORCE_PREFIXES,1.),('state',(),.1 if args.train_stage=='joint' else 1.)]:
        ps=[p for n,p in model.named_parameters() if p.requires_grad and (n.startswith(FORCE_PREFIXES) if kind=='force' else not n.startswith(FORCE_PREFIXES))]
        if ps: groups.append({'params':ps,'lr':args.lr*factor})
    optimizer=torch.optim.AdamW(groups,weight_decay=args.weight_decay)
    start=1;best=math.inf;best_epoch=0;epochs_without_improvement=0
    if args.resume_ckpt:
        optimizer.load_state_dict(source['optimizer_state_dict']);start=source['epoch']+1;best=source['best_metric']
        # Older V6 checkpoints have no early-stopping state; start their counter at zero.
        stopping=source.get('early_stopping',{})
        best_epoch=stopping.get('best_epoch',source['epoch'])
        epochs_without_improvement=stopping.get('epochs_without_improvement',0)
        r=source['rng_state'];random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch'])
        if torch.cuda.is_available() and r['cuda'] is not None: torch.cuda.set_rng_state_all(r['cuda'])
    if sets[0] is None or sets[1] is None or not len(sets[0]) or not len(sets[1]): raise ValueError('nonempty train and validation required; never use test/train to select checkpoints')
    loaders=[make_loader(d,args,shuffle=i==0) for i,d in enumerate(sets[:2])]
    if args.train_stage=='force' and not args.resume_ckpt:
        audit=run_epoch(model,loaders[1],args,cfg,spec,scaler,params,epoch=0)
        audit['audit_scope']='LF and Hybrid xyz plus calibrated Physics Fz versus HF; no Physics Fx/Fy prior exists in V5'
        audit['used_for_checkpoint_selection']=False
        json_write(directory/'baseline_audit_validation.json',audit)
    aliases={'state_teacher':'best_state_total.pt','state_student':'best_rollout.pt','force':'best_force.pt','physics':'best_physics.pt','joint':'best_joint.pt'}
    epochs_completed=start-1
    for epoch in range(start,args.epochs+1):
        if epochs_without_improvement>=args.patience: break
        training=run_epoch(model,loaders[0],args,cfg,spec,scaler,params,teacher,optimizer,epoch)
        valid=run_epoch(model,loaders[1],args,cfg,spec,scaler,params,teacher,epoch=epoch)
        record={'epoch':epoch,'train':training,'validation':valid,'learning_rates':[g['lr'] for g in optimizer.param_groups]}
        with (directory/'epochs.jsonl').open('a') as handle: handle.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
        tqdm.write(f'Epoch {epoch}: Train Loss={training["losses"]["total"]:.6g} Validation Loss={valid["losses"]["total"]:.6g} physical_score={valid["selection_score"]:.6g}')
        score=valid['selection_score']
        if score<best:
            best=score;best_epoch=epoch;epochs_without_improvement=0
            ckpt=payload(model,spec,scaler,args,cfg,params,split,epoch,best)
            ckpt['early_stopping']={'patience':args.patience,'best_epoch':best_epoch,'epochs_without_improvement':0}
            atomic_save(ckpt,directory/'best.pt');atomic_save(ckpt,directory/aliases[args.train_stage])
            json_write(directory/'best_validation_report.json',valid)
        else:
            epochs_without_improvement+=1
        last=payload(model,spec,scaler,args,cfg,params,split,epoch,best,optimizer)
        last['early_stopping']={'patience':args.patience,'best_epoch':best_epoch,'epochs_without_improvement':epochs_without_improvement}
        atomic_save(last,directory/'last_epoch.pt')
        epochs_completed=epoch
        if epochs_without_improvement>=args.patience:
            tqdm.write(f'Early stopping {args.train_stage} at epoch {epoch}: no validation selection_score improvement for {args.patience} epochs; best epoch={best_epoch}, score={best:.6g}')
            break
    best_path=directory/'best.pt'
    if not best_path.is_file(): raise RuntimeError('no best checkpoint produced')
    json_write(directory/'stage_status.json',{'stage':args.train_stage,'status':'complete','best_checkpoint':str(best_path.resolve()),'best_validation_score':best,
        'early_stopped':epochs_without_improvement>=args.patience,'epochs_completed':epochs_completed,'best_epoch':best_epoch,'patience':args.patience})
    print(f'{args.train_stage} complete: {best_path}',flush=True)


if __name__=='__main__': main()
