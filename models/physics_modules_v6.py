"""Differentiable contact geometry and feedback. No unverified rigid-body integration."""
import torch
from torch import nn
from torch.nn import functional as F


def quat_matrix(q):
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w,x,y,z = q.unbind(-1)
    return torch.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
        2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
        2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],-1).reshape(*q.shape[:-1],3,3)


def so3_exp(v):
    theta2 = v.square().sum(-1)
    theta = theta2.clamp_min(1e-12).sqrt()
    a = torch.where(theta2 < 1e-6, 1-theta2/6+theta2.square()/120, torch.sin(theta)/theta)
    b = torch.where(theta2 < 1e-6, .5-theta2/24+theta2.square()/720, (1-torch.cos(theta))/theta2.clamp_min(1e-12))
    x,y,z = v.unbind(-1); o = torch.zeros_like(x)
    k = torch.stack([o,-z,y,z,o,-x,-y,x,o],-1).reshape(*v.shape[:-1],3,3)
    eye = torch.eye(3, device=v.device, dtype=v.dtype).expand_as(k)
    return eye + a[...,None,None]*k + b[...,None,None]*(k@k)


def so3_log(r):
    skew = torch.stack([r[...,2,1]-r[...,1,2],r[...,0,2]-r[...,2,0],r[...,1,0]-r[...,0,1]],-1)*.5
    sine = skew.norm(dim=-1)
    cosine = ((r.diagonal(dim1=-2,dim2=-1).sum(-1)-1)*.5).clamp(-1,1)
    angle = torch.atan2(sine, cosine)
    regular = skew * (angle/sine.clamp_min(1e-7))[...,None]
    small = skew*(1+sine.square()[...,None]/6)
    # Axis from symmetric matrix near pi, where the skew part vanishes.
    diag = ((r.diagonal(dim1=-2,dim2=-1)+1)*.5).clamp_min(1e-12).sqrt()
    dominant = diag.argmax(-1)
    rows = []
    for i in range(3):
        axis = (r[...,i,:] + r[...,:,i])/(4*diag[...,i:i+1].clamp_min(1e-6))
        axis = axis.clone(); axis[...,i] = diag[...,i]
        rows.append(F.normalize(axis,dim=-1))
    axes = torch.stack(rows,-2)
    axis = axes.gather(-2,dominant[...,None,None].expand(*dominant.shape,1,3)).squeeze(-2)
    axis = torch.where(((axis*skew).sum(-1,keepdim=True)<0),-axis,axis)
    return torch.where((cosine < -.9999)[...,None],axis*angle[...,None],
                       torch.where((sine < 1e-4)[...,None],small,regular))


def rotation_error(pred, target):
    return so3_log(pred.transpose(-1,-2)@target).norm(dim=-1, keepdim=True)


def matrix_quaternion(r):
    """Scalar-first quaternion; choose largest diagonal branch near pi."""
    d = r.diagonal(dim1=-2,dim2=-1)
    candidates = torch.stack([1+d.sum(-1),1+d[...,0]-d[...,1]-d[...,2],
        1-d[...,0]+d[...,1]-d[...,2],1-d[...,0]-d[...,1]+d[...,2]],-1).clamp_min(1e-12).sqrt()
    w,x,y,z = candidates.unbind(-1)
    qw=torch.stack([w*w,r[...,2,1]-r[...,1,2],r[...,0,2]-r[...,2,0],r[...,1,0]-r[...,0,1]],-1)/(2*w[...,None])
    qx=torch.stack([r[...,2,1]-r[...,1,2],x*x,r[...,0,1]+r[...,1,0],r[...,0,2]+r[...,2,0]],-1)/(2*x[...,None])
    qy=torch.stack([r[...,0,2]-r[...,2,0],r[...,0,1]+r[...,1,0],y*y,r[...,1,2]+r[...,2,1]],-1)/(2*y[...,None])
    qz=torch.stack([r[...,1,0]-r[...,0,1],r[...,0,2]+r[...,2,0],r[...,1,2]+r[...,2,1],z*z],-1)/(2*z[...,None])
    allq=torch.stack([qw,qx,qy,qz],-2)
    q=allq.gather(-2,candidates.argmax(-1)[...,None,None].expand(*r.shape[:-2],1,4)).squeeze(-2)
    return F.normalize(q,dim=-1)


def contact_frame(normal, heading):
    """Columns [longitudinal, lateral, upward normal], force ON vehicle."""
    normal = F.normalize(normal, dim=-1, eps=1e-8)
    z = torch.zeros_like(normal); z[...,2] = 1
    normal = torch.where((normal.norm(dim=-1,keepdim=True)<.5),z,normal)
    x = heading-(heading*normal).sum(-1,keepdim=True)*normal
    fallback = torch.zeros_like(x); fallback[...,0] = 1
    alt = torch.zeros_like(x); alt[...,1] = 1
    fallback = torch.where((normal[...,0:1].abs()>.9),alt,fallback)
    fallback = fallback-(fallback*normal).sum(-1,keepdim=True)*normal
    x = F.normalize(torch.where((x.norm(dim=-1,keepdim=True)<1e-6),fallback,x),dim=-1)
    y = F.normalize(torch.cross(normal,x,dim=-1),dim=-1)
    return torch.stack([x,y,normal],dim=-1)


def local_to_world(force, frame): return (frame@force.unsqueeze(-1)).squeeze(-1)
def world_to_local(force, frame): return (frame.transpose(-1,-2)@force.unsqueeze(-1)).squeeze(-1)


def aggregate_contact(force_local, frame, points, reference):
    """Six-wheel resultant about an explicit reference; not chassis net external load."""
    if force_local.shape[-2:] != (6,3): raise ValueError('expected [...,6,3] in wheel order 0..5')
    world = local_to_world(force_local,frame)
    torque = torch.cross(points-reference.unsqueeze(-2),world,dim=-1)
    return world.sum(-2),torque.sum(-2),world


def generalized_contact(jacobian, world_force):
    return torch.einsum('...wic,...wi->...c',jacobian,world_force)


def contact_features(state, batch, cfg):
    """Recalculate from corrected wheel pose/velocity and current LF terrain tangent plane.

    raw layout: contact, normal xyz, LF sinkage, LF slip xy, LF relative velocity xyz,
    LF wheel center xyz, LF omega, steering angle. Terrain plane is fixed in this
    same-time refinement; no time integration or new terrain observations occur.
    """
    features,frames,points,active = {},{},{},{}
    for i in range(6):
        raw = batch[f'contact_raw_wheel{i}']
        normal = raw[...,1:4]
        wheel_rot = state['wheel_rotation'][i]
        # Wheel orientation contains continuous spin: use axle cross normal,
        # never the spinning local x axis, to construct ground tangents.
        axis = F.normalize(wheel_rot.new_tensor(cfg.spin_axis), dim=-1)
        axle = (wheel_rot@axis.view(3,1)).squeeze(-1)
        heading = torch.cross(axle,normal,dim=-1)
        frame = contact_frame(normal,heading)
        center = state['wheel_position'][i]
        displacement = center-raw[...,10:13]
        sink = (raw[...,4:5]-(displacement*normal).sum(-1,keepdim=True)).clamp_min(0)
        contact = (raw[...,0:1]>.5).to(center.dtype)
        # Current observed LF contact is authoritative for first decode. Recomputed
        # geometry may remove a contact, but never invent one without terrain data.
        if state.get('recalculate_contact',False):
            contact = contact * (sink>1e-6).to(center.dtype)
        # Ground is stationary in this dataset; wheel world velocity is therefore
        # relative to ground. Body-relative proxy velocity is not substituted.
        vel = world_to_local(state['wheel_velocity'][i],frame)
        longitudinal = cfg.wheel_radius*state['wheel_omega'][i]-vel[...,0:1]
        lateral = -vel[...,1:2]
        relpos = world_to_local(center-state['body_position'],frame)
        features[i] = torch.cat([relpos,vel,normal,sink,longitudinal,lateral,
            state['wheel_omega'][i],contact,raw[...,14:15]],-1) # 15
        frames[i],points[i],active[i] = frame,center-cfg.wheel_radius*normal,contact
    return features,frames,points,active


class ContactEdgeDecoder(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim,hidden),nn.LayerNorm(hidden),nn.GELU(),nn.Linear(hidden,3))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self,x): return self.net(x)


class DynamicsHead(nn.Module):
    """Learn a bounded state correction from *difference* contact resultant features."""
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim,hidden),nn.LayerNorm(hidden),nn.GELU(),nn.Linear(hidden,12))
        self.gate = nn.Linear(in_dim,12)
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
        nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)
    def forward(self,x):
        raw = torch.tanh(self.net(x))
        gate = torch.sigmoid(self.gate(x))
        return gate*raw,raw,gate


def refine_hierarchy(state, correction, cfg):
    dp,dv,drot,domega = correction.split(3,-1)
    dp = dp*cfg.position_scale; dv = dv*cfg.velocity_scale
    drot = drot*.05; domega = domega*cfg.angular_velocity_scale
    result = dict(state)
    new_rot = state['body_rotation']@so3_exp(drot)
    relative = new_rot@state['body_rotation'].transpose(-1,-2)
    result['body_position'] = state['body_position']+dp
    result['body_velocity'] = state['body_velocity']+dv
    result['body_rotation'] = new_rot
    result['body_angular_velocity'] = state['body_angular_velocity']+domega
    local_world=state.get('wheel_local_world',{})
    result['wheel_local_world']={i:local_to_world(v,relative) for i,v in local_world.items()}
    result['wheel_position'] = {i:p+dp+(result['wheel_local_world'][i]-local_world[i] if i in local_world else 0.) for i,p in state['wheel_position'].items()}
    result['wheel_velocity'] = {i: p+dv for i,p in state['wheel_velocity'].items()}
    result['wheel_rotation'] = {i: relative@r for i,r in state['wheel_rotation'].items()}
    if 'wheel_angular_velocity' in state:
        result['wheel_angular_velocity'] = {i:w+domega for i,w in state['wheel_angular_velocity'].items()}
    result['rockers'] = {n:{**v,'pos':v['pos']+dp,'vel':v['vel']+dv,'rot':relative@v['rot']}
                         for n,v in state.get('rockers',{}).items()}
    for v in result['rockers'].values():
        if 'angular' in v: v['angular']=v['angular']+domega
        if 'local_world' in v:
            updated=local_to_world(v['local_world'],relative)
            v['pos']=v['pos']+updated-v['local_world'];v['local_world']=updated
    result['body_roll'] = torch.atan2(new_rot[...,2,1],new_rot[...,2,2]).unsqueeze(-1)
    result['body_pitch'] = torch.asin((-new_rot[...,2,0]).clamp(-1+1e-7,1-1e-7)).unsqueeze(-1)
    return result
