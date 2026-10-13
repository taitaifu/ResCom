"""V6 shared reconstruction and loss path. Inference and training call the same code."""
from __future__ import annotations
from typing import List, Optional
import torch
from torch.nn import functional as F
from .data_utils_v6 import ROCKER_NAMES, WHEEL_IDS, input_only, flatten_group_columns
from .physics_modules_v6 import quat_matrix, so3_exp, so3_log, contact_features, world_to_local, local_to_world, rotation_error
from .load_transfer_force_v6 import load_transfer_fz_torch
from .loss_utils_v6 import LossLedger, correction_gate_loss, independent_kinematics, angular_kinematics
ASSEMBLY_SIDES = {
    "left": {"front":"lf","rear":"lm","sub":"lb","front_wheel":0,"mid_wheel":2,"rear_wheel":4},
    "right": {"front":"rf","rear":"rm","sub":"rb","front_wheel":1,"mid_wheel":3,"rear_wheel":5},
}
ASSEMBLY_DISTANCE_CONSTRAINTS = ("front_AB","front_BC","front_AC","rear_main_AD","bogie_DE","bogie_DF","middle_upright","rear_upright","middle_rear_wheel")
def indices(cols: List[str], suffixes: List[str]) -> List[int]:
    result = []
    for suffix in suffixes:
        found = next((i for i, col in enumerate(cols) if col.endswith(suffix)), None)
        if found is not None:
            result.append(found)
    return result


def omega_index(cols: List[str]) -> Optional[int]:
    for suffix in ("omega", "ang_vel_y", "ang_vel_z", "ang_vel_x"):
        found = next((i for i, col in enumerate(cols) if col.endswith(suffix)), None)
        if found is not None:
            return found
    return None


def scaler_values(scaler, key: str, device: torch.device):
    current = scaler.scalers[key]
    # Fitted statistics are immutable during training/inference. Cache their
    # device copies; checkpoint/scaler serialization only saves scaler.scalers.
    cache=getattr(scaler,'_v6_device_statistics',None)
    if cache is None: cache={};scaler._v6_device_statistics=cache
    cache_key=(key,str(device))
    sources=(current,current.mean_,current.std_,current.apply_mask_)
    entry=cache.get(cache_key)
    if entry is not None and all(old is new for old,new in zip(entry[0],sources)):
        return entry[1]
    mean = torch.as_tensor(current.mean_, device=device, dtype=torch.float32)
    std = torch.as_tensor(current.std_, device=device, dtype=torch.float32).clamp_min(1e-8)
    mask = torch.as_tensor(current.apply_mask_, device=device, dtype=torch.bool)
    values=(mean,std,mask);cache[cache_key]=(sources,values)
    return values


def inverse(x: torch.Tensor, scaler, key: str) -> torch.Tensor:
    mean, std, mask = scaler_values(scaler, key, x.device)
    y = x.float().clone()
    y[..., mask] = y[..., mask] * std[mask] + mean[mask]
    return y


def transform(x: torch.Tensor, scaler, key: str) -> torch.Tensor:
    mean, std, mask = scaler_values(scaler, key, x.device)
    y = x.float().clone()
    y[..., mask] = (y[..., mask] - mean[mask]) / std[mask]
    return y


def group_std(scaler, key: str, ix: List[int], ref: torch.Tensor) -> torch.Tensor:
    _, std, _ = scaler_values(scaler, key, ref.device)
    return std[ix].view(*([1] * (ref.ndim - 1)), -1)


def residual_raw(pred_scaled: torch.Tensor, res_cols: List[str], hf_cols: List[str], res_key: str, scaler) -> torch.Tensor:
    raw = inverse(pred_scaled, scaler, res_key)
    out = raw.new_zeros(*raw.shape[:-1], len(hf_cols))
    lookup = {name: i for i, name in enumerate(res_cols)}
    for j, col in enumerate(hf_cols):
        res_name = "res_" + col[3:] if col.startswith("hf_") else ""
        if res_name in lookup:
            out[..., j] = raw[..., lookup[res_name]]
    return out



def assembly_point(pos: torch.Tensor, rot: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
    return pos + torch.matmul(rot, local.view(local.shape[0],1,3,1)).squeeze(-1)


def assembly_rollout_loss(batch, body_pos, body_rot, rockers, wheels):
    terms=[]
    for side, cfg in ASSEMBLY_SIDES.items():
        if not all(name in rockers for name in (cfg["front"],cfg["rear"],cfg["sub"])):
            continue
        if not all(wheel in wheels for wheel in (cfg["front_wheel"],cfg["mid_wheel"],cfg["rear_wheel"])):
            continue
        key=lambda name: batch.get(f"assembly_{side}_{name}")
        meta={name:key(name) for name in ("A_front_body","A_rear_body","B_front_local","D_main_local","D_sub_local","E_local","F_local")}
        refs={name:key(f"ref_{name}") for name in ASSEMBLY_DISTANCE_CONSTRAINTS}
        if any(value is None for value in (*meta.values(),*refs.values())):
            continue
        af=assembly_point(body_pos,body_rot,meta["A_front_body"])
        ar=assembly_point(body_pos,body_rot,meta["A_rear_body"])
        bf=assembly_point(rockers[cfg["front"]]["pos"],rockers[cfg["front"]]["rot"],meta["B_front_local"])
        dm=assembly_point(rockers[cfg["rear"]]["pos"],rockers[cfg["rear"]]["rot"],meta["D_main_local"])
        ds=assembly_point(rockers[cfg["sub"]]["pos"],rockers[cfg["sub"]]["rot"],meta["D_sub_local"])
        em=assembly_point(rockers[cfg["sub"]]["pos"],rockers[cfg["sub"]]["rot"],meta["E_local"])
        fr=assembly_point(rockers[cfg["sub"]]["pos"],rockers[cfg["sub"]]["rot"],meta["F_local"])
        pairs={"front_AB":(af,bf),"front_BC":(bf,wheels[cfg["front_wheel"]]),"front_AC":(af,wheels[cfg["front_wheel"]]),"rear_main_AD":(ar,dm),"bogie_DE":(dm,em),"bogie_DF":(dm,fr),"middle_upright":(em,wheels[cfg["mid_wheel"]]),"rear_upright":(fr,wheels[cfg["rear_wheel"]]),"middle_rear_wheel":(wheels[cfg["mid_wheel"]],wheels[cfg["rear_wheel"]])}
        for name,(p0,p1) in pairs.items():
            ref=refs[name].view(refs[name].shape[0],1).clamp_min(1e-6)
            terms.append(F.smooth_l1_loss((torch.linalg.norm(p0-p1,dim=-1)-ref)/ref,torch.zeros_like(ref)))
        terms.append(F.smooth_l1_loss(dm-ds,torch.zeros_like(dm)))
    return torch.stack(terms).mean() if terms else body_pos.new_tensor(0.0)




def detach_tree(value):
    if isinstance(value,dict): return {k:detach_tree(v) for k,v in value.items()}
    return value.detach() if isinstance(value,torch.Tensor) else value


def make_step_inputs(batch,step,role):
    inputs={}
    for key in ['system','body',*ROCKER_NAMES,*[f'wheel{i}_{s}' for i in WHEEL_IDS for s in ('kin','contact')], 'physics_nodes']:
        inputs[key]=batch[f'v6_inputs_{role}_{key}'][:,step]
    for i in WHEEL_IDS:
        for name in (f'force_history_wheel{i}',f'shear_history_wheel{i}',f'contact_raw_wheel{i}'):
            inputs[name]=batch[f'v6_inputs_{name}'][:,step]
    return inputs


def _part_names(): return ['body',*[f'rocker_{s}' for s in ROCKER_NAMES],*[f'wheel{i}' for i in WHEEL_IDS]]


def state_values(state,part):
    if part=='body':
        return state['body_position'],state['body_velocity'],state['body_rotation'],state['body_angular_velocity'],None
    if part.startswith('rocker_'):
        p=state['rockers'][part[7:]]
        return p['pos'],p['vel'],p['rot'],p['angular'],None
    i=int(part[5:])
    return state['wheel_position'][i],state['wheel_velocity'][i],state['wheel_rotation'][i],state['wheel_angular_velocity'][i],state['wheel_omega'][i]


def autonomous_inputs(inputs, raw, previous, dt, spec, scaler, cfg):
    """Propagate previous predicted state, overwrite motion groups and derived proxies.

    Controls, current terrain and contact observations remain exogenous. This is
    state-autoregressive compensation with observed contact, not an unobserved
    closed-loop simulator. No subsequent LF motion enters these groups.
    """
    inputs=dict(inputs);raw={k:v.clone() for k,v in raw.items()}
    from .physics_modules_v6 import matrix_quaternion
    groups=flatten_group_columns(spec)
    for part in _part_names():
        p,v,r,w,omega=state_values(previous,part)
        p=p+v*dt; r=so3_exp(w*dt)@r
        a=raw[part].clone();a[...,:3]=p;a[...,3:6]=v;a[...,6:10]=matrix_quaternion(r);a[...,10:13]=w
        # Raw reconstruction uses the supplied SO(3) override instead of quaternions.
        if omega is not None: a[...,13:14]=omega
        raw[part]=a
        key='body' if part=='body' else part[7:] if part.startswith('rocker_') else part+'_kin'
        columns=groups[key]
        mean,std,mask=scaler_values(scaler,key,a.device)
        # Derived state features not explicitly regenerated receive neutral values.
        seq=inputs[key].clone();seq[:,-1:]=0
        prefix='lf_' if part=='body' else 'lf_susp_'+part+'_' if part.startswith('rocker_') else 'lf_'+part+'_'
        for j,c in enumerate(columns):
            suffix=c[len(prefix):] if c.startswith(prefix) else ''
            val=None
            for stem,value in (('pos_',p),('vel_',v),('ang_vel_',w)):
                if suffix in [stem+x for x in 'xyz']: val=value[..., 'xyz'.index(suffix[-1]):'xyz'.index(suffix[-1])+1]
            if suffix=='omega' and omega is not None: val=omega
            if suffix.startswith(('acc_','ang_acc_')): val=a.new_zeros(*a.shape[:-1],1)
            # Quaternion inputs are reconstructed from SO(3) using stable branch code.
            if suffix in ('q0','q1','q2','q3'):
                val=matrix_quaternion(r)[...,int(suffix[1]):int(suffix[1])+1]
            if val is not None: seq[:,-1:,j:j+1]=(val-mean[j])/std[j] if mask[j] else val
        inputs[key]=seq
        node=1 if part=='body' else 2+ROCKER_NAMES.index(part[7:]) if part.startswith('rocker_') else 2+len(ROCKER_NAMES)+int(part[5:])
        phy=inputs['physics_nodes'].clone()
        phy[:,-1:,node,:3]=p;phy[:,-1:,node,3:6]=v
        phy[:,-1:,node,6:15]=r.flatten(-2);phy[:,-1:,node,15:18]=w
        inputs['physics_nodes']=phy
    phy=inputs['physics_nodes'].clone()
    for i in WHEEL_IDS:
        motion=2+len(ROCKER_NAMES)+i;contact=motion+6
        phy[:,-1:,contact,:18]=phy[:,-1:,motion,:18]
    inputs['physics_nodes']=phy
    return inputs,raw


def reconstruct_state(model,inputs,raw,dt,spec,scaler,cfg,previous=None,role='student'):
    # Network boundary only receives the allowlisted LF/control/contact tensors.
    out=model.forward_state(input_only(inputs,role),current_index=cfg.history_len)
    body_cols=spec.target_groups.body_cols
    pidx=indices(body_cols,['pos_x','pos_y','pos_z']);vidx=indices(body_cols,['vel_x','vel_y','vel_z'])
    d=residual_raw(out['pred_res_body'],spec.res_groups.body_cols,body_cols,'res_body',scaler)
    lf=raw['body'];v=lf[...,3:6]+out['gate_body_velocity']*d[...,vidx]
    if previous is None: raise ValueError('explicit previous state required for body propagation')
    prop=previous['body_position']+.5*(previous['body_velocity']+v)*dt
    pos=prop+out['gate_body_pos']*d[...,pidx]
    lfrot=raw.get('body_rotation',quat_matrix(lf[...,6:10]))
    rot=lfrot@so3_exp(out['gate_body_att']*out['pred_delta_rot_body'])
    angular=lf[...,10:13]+out['gate_body_angular_velocity']*out['delta_body_angular_velocity']*cfg.angular_velocity_scale
    state={'z':out['z'],'body_node':out['body_node'],'wheel_kin_nodes':out['wheel_kin_nodes'],
        'wheel_contact_nodes':out['wheel_contact_nodes'],'body_position':pos,'body_velocity':v,
        'body_rotation':rot,'body_angular_velocity':angular,'body_acceleration':(v-previous['body_velocity'])/dt,
        'body_roll':torch.atan2(rot[...,2,1],rot[...,2,2]).unsqueeze(-1),
        'body_pitch':torch.asin((-rot[...,2,0]).clamp(-1+1e-7,1-1e-7)).unsqueeze(-1),
        'rockers':{},'wheel_position':{},'wheel_velocity':{},'wheel_rotation':{},'wheel_angular_velocity':{},'wheel_local_world':{},
        'wheel_omega':{},'force_base':{},'fphy_z':{},'lf_force':{},'wheel_vz':{},'prop_position':prop,
        'raw_body_position':d[...,pidx],'raw_body_velocity':d[...,vidx],'state_output':out,
        'dt':dt,'previous_body_velocity':previous['body_velocity'],'previous_wheel_position':previous['wheel_position']}
    global_delta=pos-lf[...,:3]
    body_relative=rot@lfrot.transpose(-1,-2)
    for name in ROCKER_NAMES:
        a=raw[f'rocker_{name}']; local=out[f'pred_rocker_{name}_local_delta_pos']
        partrot=raw.get(f'rocker_{name}_rotation',quat_matrix(a[...,6:10]))
        local_world=(rot@(local*out[f'gate_rocker_{name}_pos']).unsqueeze(-1)).squeeze(-1)
        state['rockers'][name]={'pos':a[...,:3]+global_delta+local_world,'local_world':local_world,
            'rot':body_relative@partrot@so3_exp(out[f'gate_rocker_{name}_att']*out[f'pred_rocker_{name}_delta_rot']),
            'vel':a[...,3:6]+out[f'delta_rocker_{name}_velocity']*cfg.velocity_scale,
            'angular':a[...,10:13]+out[f'delta_rocker_{name}_angular_velocity']*cfg.angular_velocity_scale}
    for i in WHEEL_IDS:
        cols=spec.target_groups.wheel_kin_cols[i];ix=indices(cols,['pos_x','pos_y','pos_z'])
        rd=residual_raw(out[f'pred_res_wheel{i}_kin'],spec.res_groups.wheel_kin_cols[i],cols,f'res_wheel{i}_kin',scaler)
        a=raw[f'wheel{i}'];local=rd[...,ix]
        local_world=(rot@(local*out[f'gate_wheel{i}_pos']).unsqueeze(-1)).squeeze(-1)
        p=a[...,:3]+global_delta+local_world
        state['wheel_local_world'][i]=local_world
        state['wheel_position'][i]=p
        state['wheel_velocity'][i]=a[...,3:6]+out[f'delta_wheel{i}_velocity']*cfg.velocity_scale
        state['wheel_rotation'][i]=body_relative@raw.get(f'wheel{i}_rotation',quat_matrix(a[...,6:10]))@so3_exp(out[f'gate_wheel{i}_att']*out[f'pred_wheel{i}_delta_rot'])
        state['wheel_angular_velocity'][i]=a[...,10:13]+out[f'delta_wheel{i}_angular_velocity']*cfg.angular_velocity_scale
        # Availability is fixed by training metadata, never by runtime HF input.
        supervised=getattr(model,'omega_supervised',[False]*6)[i]
        oi=next((j for j,c in enumerate(cols) if c.endswith('_omega')),None)
        delta=rd[...,oi:oi+1] if oi is not None and getattr(model,'uses_legacy_spin_head',False) else out[f'delta_wheel{i}_omega_raw']*cfg.omega_scale
        state['wheel_omega'][i]=a[...,13:14]+(out[f'gate_wheel{i}_omega']*delta if supervised else 0.)
        state['wheel_vz'][i]=(p[...,2:3]-previous['wheel_position'][i][...,2:3])/dt
        state[f'raw_wheel{i}_local']=local;state[f'raw_wheel{i}_omega']=delta
    return state


def initial_previous(batch):
    previous={'rockers':{},'wheel_position':{},'wheel_velocity':{},'wheel_rotation':{},'wheel_angular_velocity':{},'wheel_omega':{}}
    for part in _part_names():
        a=batch[f'v6_previous_lf_{part}']
        if part=='body':
            previous.update(body_position=a[...,:3],body_velocity=a[...,3:6],body_rotation=quat_matrix(a[...,6:10]),body_angular_velocity=a[...,10:13])
        elif part.startswith('rocker_'):
            previous['rockers'][part[7:]]={'pos':a[...,:3],'vel':a[...,3:6],'rot':quat_matrix(a[...,6:10]),'angular':a[...,10:13]}
        else:
            i=int(part[5:]);previous['wheel_position'][i]=a[...,:3];previous['wheel_velocity'][i]=a[...,3:6]
            previous['wheel_rotation'][i]=quat_matrix(a[...,6:10]);previous['wheel_angular_velocity'][i]=a[...,10:13];previous['wheel_omega'][i]=a[...,13:14]
    return previous


def add_force_baseline(state,inputs,lf_force,params,cfg):
    # Local contact frame applies to LF/HF labels. Physics Fz is an empirical
    # normal load regression calibrated using train HF, not analytic net force.
    _,frames,_,_=contact_features(state,inputs,cfg)
    state['load_transfer_params']=params
    for i in WHEEL_IDS:
        lf=lf_force[i]
        if cfg.input_force_frame=='world': lf=world_to_local(lf,frames[i])
        fz=load_transfer_fz_torch(state['body_roll'],state['body_pitch'],state['body_acceleration'],state['wheel_vz'][i],i,params)
        state['lf_force'][i]=lf;state['fphy_z'][i]=fz
        state['force_base'][i]=torch.cat([lf[...,:2],fz],-1)
    return state


def forward_sequence(model,batch,spec,scaler,cfg,params=None,steps=1,role='student',autonomous=False,previous=None):
    previous=initial_previous(batch) if previous is None else previous
    sequence=[];forces=[];input_history=[]
    for step in range(steps):
        inputs=make_step_inputs(batch,step,role)
        raw={part:batch[f'v6_lf_{part}'][:,step:step+1] for part in _part_names()}
        dt=batch['v6_dt'][:,step:step+1,None].clamp_min(1e-6)
        if autonomous and step>0:
            inputs,raw=autonomous_inputs(inputs,raw,previous,dt,spec,scaler,cfg)
            # Shift actual model-fed history, then append this predicted current frame.
            for key in ['system','body',*ROCKER_NAMES,*[f'wheel{i}_{s}' for i in WHEEL_IDS for s in ('kin','contact')],'physics_nodes']:
                inputs[key]=torch.cat([input_history[-1][key][:,1:],inputs[key][:,-1:]],1)
            for part in _part_names():
                _,_,r,w,_=state_values(previous,part)
                raw[part+'_rotation']=so3_exp(w*dt)@r
        state=reconstruct_state(model,inputs,raw,dt,spec,scaler,cfg,previous,role)
        if params is not None:
            lf_force={i:batch[f'v6_lf_force_wheel{i}'][:,step:step+1] for i in WHEEL_IDS}
            state=add_force_baseline(state,inputs,lf_force,params,cfg)
            state,force=model.coupled_update(state,inputs,current_index=cfg.history_len)
        else: force=None
        sequence.append(state);forces.append(force);input_history.append(inputs)
        previous=state
    return sequence,forces


def state_losses(states,batch,spec,scaler,cfg,args,ledger,metrics,prefix='state',weight=1.):
    residual=[];final=[];improve=[];gates=[];assembly=[];localreg=[];rockerreg=[];bodyreg=[];rotkin=[]
    def supervise(name,pred,truth,lf,mask,scale,tol):
        from .loss_utils_v6 import masked_mean
        scale=torch.as_tensor(scale,device=pred.device,dtype=pred.dtype).clamp_min(1e-5)
        final.append(masked_mean(F.smooth_l1_loss((pred-truth)/scale,torch.zeros_like(pred),reduction='none'),mask))
        e=(pred-truth).abs()/scale;baseline=(lf-truth).abs()/scale
        margin=args.improve_margin*(baseline.detach()>tol/scale).to(e)
        improve.append(masked_mean(F.relu(e-baseline+margin-tol/scale),mask))
        metric_scale=scale.detach().mean()
        metrics.add(name,pred,truth,lf,mask,tol,metric_scale)
        metrics.add(f'rollout/{prefix}/step_{step}/{name}',pred,truth,lf,mask,tol,metric_scale)
    for step,state in enumerate(states):
        out=state['state_output'];bodylf=batch['v6_lf_body'][:,step:step+1]
        bodyhf=batch['v6_hf_body'][:,step:step+1]
        for part in _part_names():
            predp,predv,predr,predw,predo=state_values(state,part)
            lf=batch[f'v6_lf_{part}'][:,step:step+1];hf=batch[f'v6_hf_{part}'][:,step:step+1]
            mask=batch[f'v6_hf_{part}_mask'][:,step:step+1]
            group='hf_body' if part=='body' else f'hf_{part}_kin' if part.startswith('wheel') else None
            pos_std=cfg.position_scale if group is None else group_std(scaler,group,indices(spec.target_groups.body_cols if part=='body' else spec.target_groups.wheel_kin_cols[int(part[5:])],['pos_x','pos_y','pos_z']),predp).clamp_min(1e-4)
            supervise(f'{part}/position',predp,hf[...,:3],lf[...,:3],mask[...,:3],pos_std,1e-5)
            supervise(f'{part}/velocity',predv,hf[...,3:6],lf[...,3:6],mask[...,3:6],cfg.velocity_scale,1e-5)
            supervise(f'{part}/angular_velocity',predw,hf[...,10:13],lf[...,10:13],mask[...,10:13],cfg.angular_velocity_scale,1e-5)
            truthrot=quat_matrix(hf[...,6:10]);lfrot=quat_matrix(lf[...,6:10])
            err=rotation_error(predr,truthrot);baseerr=rotation_error(lfrot,truthrot)
            supervise(f'{part}/attitude_rad',err,torch.zeros_like(err),baseerr,mask[...,6:10].amin(-1,keepdim=True),.1,1e-5)
            if part=='body':
                raw=state['raw_body_position'];gate=out['gate_body_pos'];target=hf[...,:3]-state['prop_position']
                bodyreg.append(raw.square().mean())
                gates.append(correction_gate_loss(state['raw_body_velocity'],out['gate_body_velocity'],hf[...,3:6]-lf[...,3:6],cfg.velocity_scale,mask[...,3:6]))
                gates.append(correction_gate_loss(out['delta_body_angular_velocity']*cfg.angular_velocity_scale,out['gate_body_angular_velocity'],hf[...,10:13]-lf[...,10:13],cfg.angular_velocity_scale,mask[...,10:13]))
                metrics.gate('body/velocity',state['raw_body_velocity'],out['gate_body_velocity'])
                metrics.gate('body/angular_velocity',out['delta_body_angular_velocity']*cfg.angular_velocity_scale,out['gate_body_angular_velocity'])
                att_raw=out['pred_delta_rot_body'];att_gate=out['gate_body_att']
                att_target=so3_log(lfrot.transpose(-1,-2)@truthrot)
            else:
                global_true=bodyhf[...,:3]-bodylf[...,:3]
                target=(quat_matrix(bodyhf[...,6:10]).transpose(-1,-2)@(hf[...,:3]-lf[...,:3]-global_true).unsqueeze(-1)).squeeze(-1)
                key='rocker_'+part[7:] if part.startswith('rocker_') else part
                raw=out[f'pred_{key}_local_delta_pos'] if part.startswith('rocker_') else state[f'raw_{part}_local']
                gate=out[f'gate_{key}_pos']
                (rockerreg if part.startswith('rocker_') else localreg).append((raw/pos_std).square().mean())
                body_true_relative=quat_matrix(bodyhf[...,6:10])@quat_matrix(bodylf[...,6:10]).transpose(-1,-2)
                att_raw=out[f'pred_{key}_delta_rot'];att_gate=out[f'gate_{key}_att']
                att_target=so3_log((body_true_relative@lfrot).transpose(-1,-2)@truthrot)
            gates.append(correction_gate_loss(raw,gate,target,pos_std,mask[...,:3]))
            residual.append(F.smooth_l1_loss(raw/pos_std,target/pos_std,reduction='none').mul(mask[...,:3]).sum()/mask[...,:3].sum().clamp_min(1))
            gates.append(correction_gate_loss(att_raw,att_gate,att_target,.1,mask[...,6:10].amin(-1,keepdim=True)))
            metrics.gate(part+'/position',raw,gate);metrics.gate(part+'/attitude',att_raw,att_gate)
            if predo is not None:
                supervise(f'{part}/omega',predo,hf[...,13:14],lf[...,13:14],mask[...,13:14],cfg.omega_scale,1e-4)
                i=int(part[5:]);oraw=state[f'raw_{part}_omega'];og=out[f'gate_wheel{i}_omega']
                if bool(mask[...,13:14].any()):
                    gates.append(correction_gate_loss(oraw,og,hf[...,13:14]-lf[...,13:14],cfg.omega_scale,mask[...,13:14]))
                metrics.gate(part+'/omega',oraw,og,predo-lf[...,13:14])
        assembly_error=assembly_rollout_loss(batch,state['body_position'],state['body_rotation'],state['rockers'],state['wheel_position'])
        assembly.append(assembly_error)
        metrics.add(f'rollout/{prefix}/step_{step}/assembly_error',assembly_error.reshape(1),torch.zeros_like(assembly_error).reshape(1),
            assembly_error.detach().reshape(1),scale=1.)
    mean=lambda x:torch.stack(x).mean() if x else states[0]['body_position'].sum()*0
    kin=[]
    if len(states)>1:
        dt=batch['v6_dt'][:,1:len(states)]
        for part in _part_names():
            vals=[state_values(s,part) for s in states]
            kin.append(independent_kinematics(torch.cat([v[0] for v in vals],1),torch.cat([v[1] for v in vals],1),dt,cfg.velocity_scale))
            rotkin.append(angular_kinematics(torch.cat([v[2] for v in vals],1),torch.cat([v[3] for v in vals],1),dt))
    ledger.add(prefix+'/final',mean(final),weight*args.w_state)
    ledger.add(prefix+'/residual',mean(residual),weight*args.w_state*.1)
    ledger.add(prefix+'/improve',mean(improve),weight*args.w_state*args.lambda_improve)
    ledger.add(prefix+'/gate',mean(gates),weight*args.w_gate)
    ledger.add(prefix+'/assembly',mean(assembly),weight*args.w_physics*args.lambda_assembly)
    ledger.add(prefix+'/kin',mean(kin+rotkin),weight*args.w_physics*args.lambda_kin,len(states)>1)
    ledger.add(prefix+'/wheel_local_reg',mean(localreg),weight*args.w_state*args.lambda_wheel_local_reg)
    ledger.add(prefix+'/rocker_local_reg',mean(rockerreg),weight*args.w_state*args.lambda_wheel_local_reg)
    ledger.add(prefix+'/body_delta_reg',mean(bodyreg),weight*args.w_state*args.lambda_body_delta_reg)
    if len(states)>1:
        d=torch.cat([s['raw_body_position'] for s in states],1)
        smooth=F.smooth_l1_loss(d[:,1:],d[:,:-1])
        ledger.add(prefix+'/body_delta_smooth',smooth,weight*args.w_state*args.lambda_body_delta_smooth)
        # Observed-LF rollout supervises later substeps independently of the
        # optional state-autoregressive pushforward branch.
        rollout_final=[]
        for step,s in enumerate(states[1:],1):
            for part in _part_names():
                p,v,r,w,o=state_values(s,part)
                truth=batch[f'v6_hf_{part}'][:,step:step+1]
                mask=batch[f'v6_hf_{part}_mask'][:,step:step+1]
                for pred,start,scale in ((p,0,cfg.position_scale),(v,3,cfg.velocity_scale),(w,10,cfg.angular_velocity_scale)):
                    m=mask[...,start:start+3]
                    rollout_final.append(F.smooth_l1_loss(pred/scale,truth[...,start:start+3]/scale,reduction='none').mul(m).sum()/m.sum().clamp_min(1))
                error=rotation_error(r,quat_matrix(truth[...,6:10]))/.1
                m=mask[...,6:10].amin(-1,keepdim=True)
                rollout_final.append(F.smooth_l1_loss(error,torch.zeros_like(error),reduction='none').mul(m).sum()/m.sum().clamp_min(1))
            rollout_final.append(assembly[step]*args.lambda_assembly)
        ledger.add(prefix+'/short_rollout',mean(rollout_final),weight*args.w_rollout)


def force_losses(states,forces,batch,cfg,args,ledger,metrics,prefix='force',weight=1.):
    from .loss_utils_v6 import masked_mean
    final=[];residual=[];improve=[];gates=[];reg=[];dyn=[]
    for step,(state,out) in enumerate(zip(states,forces)):
        for i in WHEEL_IDS:
            target=batch[f'v6_hf_force_wheel{i}'][:,step:step+1]
            mask=batch[f'v6_hf_force_wheel{i}_mask'][:,step:step+1]
            frame=out[f'contact_frame_wheel{i}']
            if cfg.input_force_frame=='world':
                target=world_to_local(target,frame)
                mask=mask.amin(-1,keepdim=True).expand_as(mask)
            pred=out[f'force_pred_wheel{i}'];base=state['force_base'][i];lf=state['lf_force'][i]
            raw=out[f'delta_force_wheel{i}'];gate=out[f'gate_force_wheel{i}']
            scale=cfg.force_scale
            final.append(masked_mean(F.smooth_l1_loss((pred-target)/scale,torch.zeros_like(pred),reduction='none'),mask))
            residual.append(masked_mean(F.smooth_l1_loss(raw/scale,(target-base)/scale,reduction='none'),mask*out[f'contact_active_wheel{i}']))
            improve.append(masked_mean(F.relu(((pred-target).abs()-(lf-target).abs())/scale+args.improve_margin*((lf-target).abs()>1e-3)-1e-3/scale),mask))
            reg.append((raw/scale).square().mean())
            if cfg.enable_force_gate: gates.append(correction_gate_loss(raw,gate,target-base,scale,mask*out[f'contact_active_wheel{i}']))
            metrics.gate(f'wheel{i}/force',raw,gate,out[f'effective_force_wheel{i}'])
            for j,a in enumerate(('Fx','Fy','Fz')):
                metrics.add(f'wheel{i}/force_{a}',pred[...,j:j+1],target[...,j:j+1],lf[...,j:j+1],mask[...,j:j+1],1e-3,scale)
                metrics.add(f'wheel{i}/baseline_hybrid_{a}',base[...,j:j+1],target[...,j:j+1],lf[...,j:j+1],mask[...,j:j+1],1e-3,scale)
                if a=='Fz': metrics.add(f'wheel{i}/baseline_physics_Fz',state['fphy_z'][i],target[...,2:3],lf[...,2:3],mask[...,2:3],1e-3,scale)
                metrics.add(f'rollout/{prefix}/step_{step}/wheel{i}/force_{a}',pred[...,j:j+1],target[...,j:j+1],lf[...,j:j+1],mask[...,j:j+1],1e-3,scale)
        if args.lambda_dyn>0:
            if not cfg.dynamics_calibrated: raise ValueError('dyn_loss requires calibrated aggregate COM/inertia metadata')
            # Difference formulation: gravity cancels. This is explicitly disabled
            # until aggregate COM and whole-vehicle inertias have been calibrated.
            from .physics_modules_v6 import aggregate_contact
            f=torch.stack([out[f'force_pred_wheel{i}']-state['lf_force'][i] for i in WHEEL_IDS],-2)
            frame=torch.stack([out[f'contact_frame_wheel{i}'] for i in WHEEL_IDS],-3)
            points=torch.stack([out[f'contact_point_wheel{i}'] for i in WHEEL_IDS],-2)
            df,torque,_=aggregate_contact(f,frame,points,state['body_position'])
            if step>0:
                dt=batch['v6_dt'][:,step:step+1,None]
                dv=state['body_velocity']-batch['v6_lf_body'][:,step:step+1,3:6]
                dvprev=states[step-1]['body_velocity']-batch['v6_lf_body'][:,step-1:step,3:6]
                dyn.append(F.smooth_l1_loss((dv-dvprev)/dt,df/cfg.mass))
                # Positive whole-vehicle principal inertias and COM are explicit
                # prerequisites. Compute gyroscopic terms in each body frame.
                r=state['body_rotation'];rp=states[step-1]['body_rotation']
                w=world_to_local(state['body_angular_velocity'],r)
                wp=world_to_local(states[step-1]['body_angular_velocity'],rp)
                lf_r=quat_matrix(batch['v6_lf_body'][:,step:step+1,6:10])
                lf_rp=quat_matrix(batch['v6_lf_body'][:,step-1:step,6:10])
                lf_w=world_to_local(batch['v6_lf_body'][:,step:step+1,10:13],lf_r)
                lf_wp=world_to_local(batch['v6_lf_body'][:,step-1:step,10:13],lf_rp)
                inertia=w.new_tensor(cfg.inertia_body)
                predicted_load=local_to_world(inertia*(w-wp)/dt+torch.cross(w,inertia*w,dim=-1),r)
                baseline_load=local_to_world(inertia*(lf_w-lf_wp)/dt+torch.cross(lf_w,inertia*lf_w,dim=-1),lf_r)
                dyn.append(F.smooth_l1_loss((predicted_load-baseline_load)/cfg.torque_scale,torque/cfg.torque_scale))
    mean=lambda x:torch.stack(x).mean() if x else states[0]['body_velocity'].sum()*0
    ledger.add(prefix+'/final',mean(final),weight*args.w_force)
    ledger.add(prefix+'/residual',mean(residual),weight*args.w_force*.1)
    ledger.add(prefix+'/improve',mean(improve),weight*args.w_force*args.lambda_improve)
    ledger.add(prefix+'/gate',mean(gates),weight*args.w_gate,cfg.enable_force_gate)
    ledger.add(prefix+'/residual_reg',mean(reg),weight*args.w_force*args.lambda_force_residual_reg)
    ledger.add(prefix+'/short_rollout',mean(final[6:]),weight*args.w_rollout,len(states)>1)
    ledger.add(prefix+'/dyn',mean(dyn),weight*args.w_physics*args.lambda_dyn,args.lambda_dyn>0)
    # Baseline ablation still records an explicit disabled dynamics entry.
    if cfg.enable_dynamics_update:
        ledger.add(prefix+'/dynamics_correction',mean([(s['body_velocity']-s['initial_state']['body_velocity']).square().mean() for s in states]),weight*args.w_physics*1e-4)
