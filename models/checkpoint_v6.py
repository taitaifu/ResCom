"""Explicit checkpoint architecture and feature compatibility checks."""
from pathlib import Path
import json
import numpy as np
import torch
from .config_v6 import V6Config
from .data_utils_v6 import get_group_dims, GroupStandardizer, NumpyStandardScaler, spec_from_dict
from .graph_temporal_hgt_compensation_v6 import HGTGraphTemporalCompensationModelV6
from .runtime_v6 import indices, omega_index

ARCH_KEYS=('hidden_dim','graph_layers','tcn_dim','lstm_dim','lstm_layers','dropout','force_hidden_dim')
NEW_PREFIXES=('physics_edge_mlps.','physics_attention_mlps.','enhanced_history.','contact_edge_decoder.',
    'force_gate_head.','dynamics_head.','body_angular_velocity_head.','body_velocity_gate_head.',
    'body_angular_gate_head.','wheel_spin_head.','part_velocity_head.','part_angular_head.')


def physics_resource_fingerprints(cfg):
    import hashlib
    from .data_utils_v6 import DEFAULT_VEHICLE_CONFIG
    return {name:hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else 'absent: documented default'
            for name,path in [('vehicle_geometry',DEFAULT_VEHICLE_CONFIG),('terrain',Path(cfg.terrain_file))]}


def read_checkpoint(path,device='cpu'):
    return torch.load(path,map_location=device,weights_only=False)


def restore_scaler(ckpt):
    if not ckpt.get('scaler_state'): raise ValueError('checkpoint must contain scaler_state')
    scaler=GroupStandardizer()
    scaler.scalers={k:NumpyStandardScaler.from_dict(v) for k,v in ckpt['scaler_state'].items()}
    return scaler


def build_model(spec,args,cfg,role='student'):
    body=spec.res_groups.body_cols;wheel=spec.res_groups.wheel_kin_cols[0]
    pos=indices(body,['pos_x','pos_y','pos_z']);vel=indices(body,['vel_x','vel_y','vel_z'])
    wp=indices(wheel,['pos_x','pos_y','pos_z']);wi=omega_index(wheel)
    if any(len(x)!=3 for x in (pos,vel,wp)): raise ValueError('body position/velocity and wheel position xyz required')
    for i in range(6):
        cols=spec.res_groups.wheel_kin_cols[i]
        if len(cols)!=len(wheel) or indices(cols,['pos_x','pos_y','pos_z'])!=wp or omega_index(cols)!=wi:
            raise ValueError(f'inconsistent shared wheel residual layout: wheel{i}')
    val=lambda k,d:getattr(args,k,d) if not isinstance(args,dict) else args.get(k,d)
    model=HGTGraphTemporalCompensationModelV6(get_group_dims(spec),node_hidden_dim=val('hidden_dim',192),
        graph_layers=val('graph_layers',3),tcn_hidden_dim=val('tcn_dim',256),lstm_hidden_dim=val('lstm_dim',192),
        lstm_layers=val('lstm_layers',2),dropout=val('dropout',.1),force_hidden_dim=val('force_hidden_dim',48),
        body_pos_indices=pos,body_vel_indices=vel,wheel_pos_indices=wp,wheel_omega_index=wi,config=cfg,role=role)
    model.uses_legacy_spin_head=any(c.endswith('_omega') for c in wheel) and any(c.endswith('_omega') for c in spec.target_groups.wheel_kin_cols[0])
    return model


def verify_features(ckpt,spec,scaler=None,allow_v5=False):
    if 'column_spec' not in ckpt: raise ValueError('checkpoint lacks feature specification; transfer refused')
    old=spec_from_dict(ckpt['column_spec']).to_dict();new=spec.to_dict()
    if old!=new: raise ValueError('checkpoint column ordering/grouping differs; transfer refused')
    if scaler is not None:
        for k,d in ckpt['scaler_state'].items():
            if k not in scaler.scalers: raise ValueError(f'missing scaler {k}')
            current=scaler.scalers[k].to_dict()
            for field in ('mean','std','apply_mask'):
                # Stored scaler field spelling is checked through the actual payload.
                source_key=field+'_' if field+'_' in d else field
                if source_key in d and (np.shape(d[source_key])!=np.shape(current[source_key]) or
                    not np.allclose(d[source_key],current[source_key],rtol=1e-5,atol=1e-6)):
                    raise ValueError(f'{k}/{source_key}: incompatible normalization')


def compatible_load(model,ckpt,teacher_to_student=False,report_path=None):
    source=ckpt['model_state_dict'];current=model.state_dict()
    skipped_prefix=('temporal_net.',) if teacher_to_student else ()
    semantic_skip=()
    semantic_mismatch={}
    if ckpt.get('version')!=6:
        old=spec_from_dict(ckpt['column_spec'])
        if not any(c.endswith('_omega') for c in old.target_groups.wheel_kin_cols[0]):
            # Matching tensor shapes are insufficient: V5's missing output spin
            # led to zero/world-component omega in the force decoder input.
            semantic_skip=('force_residual_head.',)
            semantic_mismatch['force_residual_head.']='independently initialized: canonical LF axle spin replaces V5 zero/world-component omega input'
    loaded={};shape_mismatch={};unexpected=[]
    for k,v in source.items():
        if k.startswith(semantic_skip): continue
        if k.startswith(skipped_prefix):
            shape_mismatch[k]={'source':list(v.shape),'target':list(current[k].shape) if k in current else None,
                               'policy':'teacher bidirectional LSTM is not transferable to causal Student'}
        elif k not in current: unexpected.append(k)
        elif current[k].shape!=v.shape:
            shape_mismatch[k]={'source':list(v.shape),'target':list(current[k].shape),'policy':'rejected'}
        else: loaded[k]=v
    missing=[k for k in current if k not in loaded]
    report={'loaded':sorted(loaded),'missing':missing,'unexpected':unexpected,'shape_mismatch':shape_mismatch,'semantic_mismatch':semantic_mismatch}
    print(json.dumps(report,indent=2),flush=True)
    if report_path: Path(report_path).write_text(json.dumps(report,indent=2)+'\n')
    bad=[k for k,v in shape_mismatch.items() if v['policy']=='rejected']
    critical=[k for k in missing if not k.startswith(NEW_PREFIXES+skipped_prefix+semantic_skip)]
    if bad or critical or unexpected:
        raise ValueError(f'checkpoint transfer rejected: shape_mismatch={bad}, critical_missing={critical}, unexpected={unexpected}')
    # Only exact keys+shapes with validated features enter the destination.
    model.load_state_dict(loaded,strict=False)
    return report


def restore_model(ckpt,device='cpu'):
    if ckpt.get('version')!=6: raise ValueError('inference requires a V6 checkpoint')
    cfg=V6Config.from_dict(ckpt['v6_config']);spec=spec_from_dict(ckpt['column_spec'])
    expected=ckpt['feature_config'].get('physics_resource_fingerprints')
    if expected is not None and expected!=physics_resource_fingerprints(cfg):
        raise ValueError('terrain/vehicle geometry differs from checkpoint physics configuration')
    for name,value in [('physics_node_dim',32),('contact_dim',15),('legacy_contact_history_dim',7),('enhanced_shear_history_dim',8)]:
        if ckpt['feature_config'][name]!=value: raise ValueError(f'feature schema mismatch: {name}')
    role=ckpt['role'];model=build_model(spec,ckpt['architecture'],cfg,role)
    model.omega_supervised=ckpt['feature_config']['omega_supervised']
    model.load_state_dict(ckpt['model_state_dict'],strict=True)
    return model.to(device),spec,restore_scaler(ckpt),cfg


def atomic_save(payload,path):
    import os
    target=Path(path);target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_suffix(target.suffix+'.tmp')
    torch.save(payload,temporary);os.replace(temporary,target)
