"""Masked physical-unit metrics, conservative loss accounting and gate targets."""
import numpy as np
import torch
from torch.nn import functional as F
from .physics_modules_v6 import so3_exp, so3_log


def masked_mean(value, mask):
    mask = mask.to(value).expand_as(value)
    return (value*mask).sum()/mask.sum().clamp_min(1)


def correction_gate_loss(raw, gate, true_delta, scale, mask=None):
    """Optimal signed correction, actual harm, and undercorrection on useful axes.

    Labels are detached; gradients only go to predicted residual/gate. Tiny LF
    errors prefer no correction, while useful small raw predictions get a high
    gate target (not a residual-magnitude based suppression target).
    """
    scale = torch.as_tensor(scale,device=raw.device,dtype=raw.dtype).clamp_min(1e-6)
    mask = torch.ones_like(raw) if mask is None else mask
    r = raw/scale; t = true_delta/scale
    optimum = (r.detach()*t.detach()/(r.detach().square()+1e-10)).clamp(0,1)
    # A zero-initialized decoder must not teach its gate to close before the
    # residual can learn. Corrective direction remains decisive once nonzero.
    optimum = torch.where((r.detach().abs()<1e-6)&(t.detach().abs()>1e-4),torch.full_like(optimum,.75),optimum)
    optimum = torch.where(t.detach().abs()>1e-4,optimum,torch.zeros_like(optimum))
    label = masked_mean(F.smooth_l1_loss(gate,optimum,reduction='none'),mask)
    effective = gate*r
    before=t.abs(); after=(t-effective).abs()
    harm = F.relu(after-before-1e-4)
    useful = ((r.detach()*t.detach()>0)&(t.detach().abs()>1e-4)).to(r)
    desired = optimum*r.detach()
    amplitude = (effective-desired).square()*useful
    return label + .25*masked_mean(harm,mask) + .1*masked_mean(amplitude,mask)


def independent_kinematics(position, velocity, dt, scale=1.):
    if position.shape[1]<2: return position.sum()*0
    d = dt.reshape(dt.shape[0],-1,1).clamp_min(1e-6)
    fd=(position[:,1:]-position[:,:-1])/d
    independent=.5*(velocity[:,1:]+velocity[:,:-1])
    return F.smooth_l1_loss(fd/scale,independent/scale)


def angular_kinematics(rotation, angular_velocity, dt, scale=1.):
    if rotation.shape[1]<2: return rotation.sum()*0
    # World angular velocity; relative rotation left-multiplies world orientation.
    step=.5*(angular_velocity[:,1:]+angular_velocity[:,:-1])*dt[...,None]
    predicted_next=so3_exp(step)@rotation[:,:-1]
    discrepancy=so3_log(predicted_next.transpose(-1,-2)@rotation[:,1:])
    return F.smooth_l1_loss(discrepancy/scale,torch.zeros_like(discrepancy))


class LossLedger:
    def __init__(self, reference): self.reference=reference;self.entries={}
    def add(self,name,raw,weight,enabled=True):
        if name in self.entries: raise ValueError(f'duplicate loss {name}')
        if not enabled: raw=self.reference.sum()*0
        self.entries[name]=(raw,float(weight) if enabled else 0.,bool(enabled))
    @property
    def total(self):
        return sum((r*w for r,w,_ in self.entries.values()),self.reference.sum()*0)
    def scalars(self):
        result={};names=list(self.entries)
        # One device transfer for the complete batch log, rather than two
        # CUDA synchronizations for every individual loss.
        packed=torch.stack([self.entries[name][0].detach().reshape(()) for name in names]+[self.total.detach().reshape(())]).cpu().tolist()
        for name,value in zip(names,packed):
            _,w,on=self.entries[name]
            result[f'loss/{name}/raw']=value
            result[f'loss/{name}/weighted']=value*w
            result[f'loss/{name}/enabled']=on
            result[f'loss/{name}/weight']=w
        result['total']=packed[-1]
        return result
    def gradients(self,parameters):
        """Per-loss effective gradient norm before backward, without touching .grad."""
        params=[p for p in parameters if p.requires_grad]
        result={}
        for name,(raw,w,on) in self.entries.items():
            norm=0.
            if on and w and raw.requires_grad and params:
                grad=torch.autograd.grad(raw*w,params,retain_graph=True,allow_unused=True)
                squares=[g.detach().float().square().sum() for g in grad if g is not None]
                if squares: norm=float(torch.stack(squares).sum().sqrt())
            result[f'loss/{name}/effective_gradient']=norm
        return result


class PhysicalMetrics:
    """Accumulate samples, not means of batch P95s. Tolerances in physical units."""
    def __init__(self,defer_transfer=False):
        self.values={};self.gates={};self.selection={};self.running_mae={}
        self.defer_transfer=defer_transfer;self.pending=[]
    def add(self,name,pred,truth,lf,mask=None,tolerance=1e-6,scale=1.):
        if self.defer_transfer:
            parts=[x.detach().float().expand_as(pred).reshape(-1) for x in (pred,truth,lf)]
            parts.append(torch.ones_like(pred).reshape(-1) if mask is None else mask.detach().float().expand_as(pred).reshape(-1))
            fixed_scale=None
            if isinstance(scale,torch.Tensor):
                parts.append(scale.detach().to(device=pred.device,dtype=torch.float32).reshape(1))
            else: fixed_scale=float(scale)
            self.pending.append(('value',name,parts,(tolerance,fixed_scale)))
            return
        def array(x): return x.detach().float().cpu().numpy().reshape(-1)
        p,t,b=map(array,(pred,truth,lf))
        m=None if mask is None else np.broadcast_to(mask.detach().cpu().numpy(),pred.shape).reshape(-1)
        self._record_value(name,p,t,b,m,tolerance,float(scale))
    def _record_value(self,name,p,t,b,mask,tolerance,scale):
        valid=np.isfinite(p)&np.isfinite(t)&np.isfinite(b)
        if mask is not None: valid &= mask>.5
        if not valid.any(): return
        e=np.abs(p[valid]-t[valid]);base=np.abs(b[valid]-t[valid])
        self.values.setdefault(name,[]).append((e,base,tolerance,scale))
        total,count=self.running_mae.get(name,(0.,0))
        self.running_mae[name]=(total+float(e.sum()),count+len(e))
    def gate(self,name,raw,gate,effective=None):
        if self.defer_transfer:
            parts=[raw.detach().float().reshape(-1),gate.detach().float().reshape(-1)]
            if effective is not None: parts.append(effective.detach().float().reshape(-1))
            self.pending.append(('gate',name,parts,None))
            return
        a=raw.detach().float().cpu().numpy().reshape(-1);g=gate.detach().float().cpu().numpy().reshape(-1)
        e=a*g if effective is None else effective.detach().float().cpu().numpy().reshape(-1)
        self.gates.setdefault(name,[]).append((a,g,e))
    def flush(self):
        """Transfer all detached batch metrics together, preserving exact samples.

        Drain each device separately. CPU calculation of errors, masks, P95 and
        ratios is identical to the eager path; no metric sampling is introduced.
        """
        if not self.pending: return
        grouped={}
        for item in self.pending: grouped.setdefault(item[2][0].device,[]).append(item)
        for items in grouped.values():
            packed=torch.cat([part for item in items for part in item[2]]).cpu().numpy()
            offset=0
            for kind,name,parts,metadata in items:
                arrays=[]
                for part in parts:
                    size=part.numel();arrays.append(packed[offset:offset+size]);offset+=size
                if kind=='value':
                    p,t,b,m=arrays[:4]
                    tolerance,fixed_scale=metadata
                    scale=fixed_scale if fixed_scale is not None else float(arrays[4][0])
                    self._record_value(name,p,t,b,m,tolerance,scale)
                else:
                    a,g=arrays[:2];e=arrays[2] if len(arrays)==3 else a*g
                    # Copies prevent a tiny gate view retaining the entire
                    # packed batch buffer throughout a long epoch.
                    self.gates.setdefault(name,[]).append((a.copy(),g.copy(),e.copy()))
        self.pending.clear()
    def quick_mae(self,name):
        self.flush()
        total,count=self.running_mae.get(name,(0.,0))
        return total/max(count,1)
    def summarize(self):
        self.flush()
        result={};scores=[];harms=[]
        for name,items in self.values.items():
            e=np.concatenate([x[0] for x in items]);b=np.concatenate([x[1] for x in items])
            tol,scale=items[0][2:]
            mae=float(e.mean());bm=float(b.mean())
            ratio=None if bm<=tol else float((bm-mae)/bm)
            harmful=float((e>b+tol).mean())
            result[name]={'MAE':mae,'RMSE':float(np.sqrt(np.mean(e*e))),'P95':float(np.percentile(e,95)),
                'LF_MAE':bm,'LF_RMSE':float(np.sqrt(np.mean(b*b))),'LF_P95':float(np.percentile(b,95)),
                'Corrected_Error':mae,'LF_Baseline_Error':bm,'Improvement_Ratio':ratio,
                'Improved_Fraction':float((e<b-tol).mean()),'Harmful_Ratio':harmful,
                'Absolute_Improvement':bm-mae,'Near_Zero_LF_Fraction':float((b<=tol).mean()),'count':len(e)}
            if name.startswith(('body/','rocker_','wheel')) and '/baseline_' not in name and not name.startswith('rollout/'):
                scores.append(mae/max(scale,1e-8));harms.append(harmful)
        for name,items in self.gates.items():
            a=np.concatenate([x[0] for x in items]);g=np.concatenate([x[1] for x in items])
            e=np.concatenate([x[2] for x in items])
            result['gate/'+name]={'raw_residual':float(np.abs(a).mean()),'gate_value':float(g.mean()),
                'effective_residual':float(np.abs(e).mean())}
        result['selection_score']=float(np.mean(scores)+.05*np.mean(harms)) if scores else None
        rollout={}
        for name,value in result.items():
            if not name.startswith('rollout/'): continue
            parts=name.split('/')
            if len(parts)<4 or not parts[2].startswith('step_'): continue
            mode=parts[1];step=int(parts[2][5:]);kind='/'.join(parts[3:])
            rollout.setdefault(mode,{}).setdefault(kind,{})[step]=value['MAE']
        result['rollout_error_accumulation']={mode:{kind:{'first_step_error':steps[min(steps)],
            'last_step_error':steps[max(steps)],'absolute_growth':steps[max(steps)]-steps[min(steps)],
            'per_step':steps} for kind,steps in kinds.items()} for mode,kinds in rollout.items()}
        return result
