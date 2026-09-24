from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass(frozen=True)
class TerramechanicsParams:
    r: float = 0.135
    b: float = 0.16
    h: float = 0.02
    c1: float = 0.4
    c2: float = 0.15
    c3: float = -0.2
    fn_max: float = 2000.0
    fn_ave: float = 350.0
    c_T1: float = 0.2
    c_d1: float = 0.0
    c_d2: float = 0.2
    c_d3: float = 0.1
    first_damp_coef: float = 250.0
    second_damp_coef: float = 650.0
    sinkage_max: float = 0.08
    contact_threshold: float = 5e-5
    eps: float = 1e-6
    acos_eps: float = 1e-4
    pow_eps: float = 1e-8
    max_exp_arg: float = 50.0
    max_tan_angle: float = 1.45


def _clamp(x: torch.Tensor, lo: float | torch.Tensor, hi: float | torch.Tensor) -> torch.Tensor:
    return torch.maximum(torch.minimum(x, torch.as_tensor(hi, dtype=x.dtype, device=x.device)),
                         torch.as_tensor(lo, dtype=x.dtype, device=x.device))


def _safe_acos(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    return torch.acos(x.clamp(-1.0 + eps, 1.0 - eps))


def _safe_den(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    sign = torch.where(x < 0.0, -torch.ones_like(x), torch.ones_like(x))
    return torch.where(torch.abs(x) < eps, sign * eps, x)


def _safe_div(num: torch.Tensor, den: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return num / _safe_den(den, eps)


def _safe_pow(base: torch.Tensor, exponent: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return base.clamp_min(eps).pow(exponent)


def _safe_tan(angle: torch.Tensor, max_angle: float = 1.45) -> torch.Tensor:
    return torch.tan(angle.clamp(-max_angle, max_angle))


def _safe_exp_neg(x: torch.Tensor, max_arg: float = 50.0) -> torch.Tensor:
    return torch.exp((-x).clamp(-max_arg, max_arg))


def _finite(x: torch.Tensor, limit: float = 1.0e6) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=limit, neginf=-limit).clamp(-limit, limit)


def calculate_slip(omega: torch.Tensor, v_long: torch.Tensor, r: float = 0.135) -> torch.Tensor:
    rs = torch.as_tensor(r, dtype=omega.dtype, device=omega.device)
    del_v = torch.where(omega > 0.0, rs * omega - v_long, torch.abs(rs * omega) - torch.abs(v_long))
    small = (torch.abs(del_v) <= 1e-3) | (torch.abs(omega) < 1e-3)
    reverse = omega * v_long < 0.0
    drive = del_v > 1e-3
    s_drive = 1.0 - torch.abs(_safe_div(v_long, rs * omega, 1e-6))
    s_brake = _safe_div(torch.abs(rs * omega), torch.abs(v_long), 1e-6) - 1.0
    s = torch.where(small, torch.zeros_like(omega), torch.where(reverse, torch.ones_like(omega), torch.where(drive, s_drive, s_brake)))
    return s.clamp(-1.0, 1.0)


def calculate_beta(v_long: torch.Tensor, v_lat: torch.Tensor) -> torch.Tensor:
    beta = torch.pi / 2.0 - torch.atan2(v_long, v_lat)
    beta = torch.where(beta > torch.pi / 2.0, beta - torch.pi, beta)
    beta = torch.where(torch.abs(v_lat) <= 1e-3, torch.zeros_like(beta), beta)
    beta = torch.where(torch.abs(beta) < 1e-2, torch.zeros_like(beta), beta)
    return beta


def correct_sinkage(
    sinkage_lf: torch.Tensor,
    wheel_z_pred: torch.Tensor,
    wheel_z_lf: torch.Tensor,
    sinkage_max: float = 0.08,
) -> torch.Tensor:
    return (sinkage_lf - (wheel_z_pred - wheel_z_lf)).clamp(0.0, sinkage_max)


def calculate_sinkage_from_terrain(
    wheel_z_pred: torch.Tensor,
    terrain_z: torch.Tensor,
    r: float = 0.135,
    sinkage_max: float = 0.08,
) -> torch.Tensor:
    radius = torch.as_tensor(r, dtype=wheel_z_pred.dtype, device=wheel_z_pred.device)
    terrain_z = terrain_z.to(device=wheel_z_pred.device, dtype=wheel_z_pred.dtype)
    return (radius - (wheel_z_pred - terrain_z)).clamp(0.0, sinkage_max)


def calculate_sinkage_from_contact_plane(
    wheel_pos_pred: torch.Tensor,
    contact_point: torch.Tensor,
    contact_normal: torch.Tensor,
    r: float = 0.135,
    sinkage_max: float = 0.08,
    eps: float = 1e-6,
) -> torch.Tensor:
    radius = torch.as_tensor(r, dtype=wheel_pos_pred.dtype, device=wheel_pos_pred.device)
    point = contact_point.to(device=wheel_pos_pred.device, dtype=wheel_pos_pred.dtype)
    normal = contact_normal.to(device=wheel_pos_pred.device, dtype=wheel_pos_pred.dtype)
    normal_norm = torch.linalg.norm(normal, dim=-1).clamp_min(eps)
    projected_height = ((wheel_pos_pred - point) * normal).sum(dim=-1) / normal_norm
    return (radius - projected_height).clamp(0.0, sinkage_max)


def _terrain_param(contact_features: Optional[Dict[str, torch.Tensor]], name: str, ref: torch.Tensor, default: float) -> torch.Tensor:
    if contact_features is not None and name in contact_features:
        return contact_features[name].to(device=ref.device, dtype=ref.dtype)
    return torch.full_like(ref, float(default))


def _terrain_height(terrain_params: Optional[Dict[str, torch.Tensor]], ref: torch.Tensor) -> Optional[torch.Tensor]:
    if terrain_params is None:
        return None
    for name in ("terrain_z", "z"):
        if name in terrain_params:
            return terrain_params[name].to(device=ref.device, dtype=ref.dtype)
    return None


def _contact_plane(terrain_params: Optional[Dict[str, torch.Tensor]], ref: torch.Tensor) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if terrain_params is None:
        return None, None
    point = terrain_params.get("contact_point")
    normal = terrain_params.get("contact_normal")
    if point is None or normal is None:
        return None, None
    return point.to(device=ref.device, dtype=ref.dtype), normal.to(device=ref.device, dtype=ref.dtype)


def wheel_terrain_force(
    omega: torch.Tensor,
    body_velocity: torch.Tensor,
    wheel_z_pred: torch.Tensor,
    wheel_z_lf: torch.Tensor,
    sinkage_lf: torch.Tensor,
    wheel_pos_pred: Optional[torch.Tensor] = None,
    in_contact: Optional[torch.Tensor] = None,
    terrain_params: Optional[Dict[str, torch.Tensor]] = None,
    params: TerramechanicsParams = TerramechanicsParams(),
) -> Dict[str, torch.Tensor]:
    """Differentiable Fx/Fy/Fz approximation of TerramechanicsRigid.

    Inputs are tensors with a common leading shape. ``body_velocity`` ends in xyz.
    Forces are returned in the same local/world-aligned frame as the data columns.
    """
    vx = body_velocity[..., 0]
    vy = body_velocity[..., 1] if body_velocity.shape[-1] > 1 else torch.zeros_like(vx)
    vz = body_velocity[..., 2] if body_velocity.shape[-1] > 2 else torch.zeros_like(vx)

    slip = calculate_slip(omega, vx, params.r)
    beta = calculate_beta(vx, vy)
    contact_point, contact_normal = _contact_plane(terrain_params, wheel_z_pred)
    if wheel_pos_pred is not None and contact_point is not None and contact_normal is not None:
        sinkage = calculate_sinkage_from_contact_plane(
            wheel_pos_pred,
            contact_point,
            contact_normal,
            params.r,
            params.sinkage_max,
            params.eps,
        )
    else:
        terrain_z = _terrain_height(terrain_params, wheel_z_pred)
        if terrain_z is not None:
            sinkage = calculate_sinkage_from_terrain(wheel_z_pred, terrain_z, params.r, params.sinkage_max)
        else:
            sinkage = correct_sinkage(sinkage_lf, wheel_z_pred, wheel_z_lf, params.sinkage_max)
    effective_sinkage = (sinkage - params.contact_threshold).clamp_min(0.0)

    theta1 = _safe_acos((params.r - effective_sinkage) / params.r, params.acos_eps)
    theta2 = params.c3 * theta1
    thetam = (params.c1 + params.c2 * slip) * theta1
    n0 = _terrain_param(terrain_params, "n0", vx, 0.79)
    n1 = _terrain_param(terrain_params, "n1", vx, 0.70)
    n = (n0 + n1 * torch.abs(slip)).clamp(0.2, 3.0)
    if terrain_params is not None and "n" in terrain_params and "n0" not in terrain_params:
        n = _terrain_param(terrain_params, "n", vx, 1.0).clamp(0.2, 3.0)
    kc = _terrain_param(terrain_params, "Kc", vx, -20700.0).clamp(-1.0e7, 1.0e7)
    kphi = _terrain_param(terrain_params, "Kphi", vx, 1594800.0).clamp_min(0.0)
    cohesion = _terrain_param(terrain_params, "c", vx, 460.0).clamp_min(0.0)
    phi = _terrain_param(terrain_params, "phi", vx, 0.61).clamp(-params.max_tan_angle, params.max_tan_angle)
    shear_k = _terrain_param(terrain_params, "K", vx, 0.0133).clamp_min(params.eps)

    theta_m2 = (thetam - theta2).clamp_min(params.eps)
    theta_1m = (theta1 - thetam).clamp_min(params.eps)
    theta_12 = (theta1 - theta2).clamp_min(params.eps)
    a1 = _safe_div(torch.cos(thetam) - torch.cos(theta2), theta_m2, params.eps)
    a2 = _safe_div(torch.cos(thetam) - torch.cos(theta1), theta_1m, params.eps)
    b1 = _safe_div(torch.sin(thetam) - torch.sin(theta2), theta_m2, params.eps)
    b2 = _safe_div(torch.sin(thetam) - torch.sin(theta1), theta_1m, params.eps)
    aa = a1 + a2
    bb = b1 + b2
    cc = theta_12 / 2.0

    contact_depth = (torch.cos(thetam) - torch.cos(theta1)).clamp_min(params.pow_eps)
    sigma_m = (kc / params.b + kphi) * (params.r ** n) * _safe_pow(contact_depth, n, params.pow_eps)
    fn = (params.r * params.b * aa * sigma_m).clamp(0.0, params.fn_max)

    rs = params.r + 0.5 * params.h
    rj = torch.where(slip < 0.15, torch.full_like(slip, params.r),
                     torch.where(slip > 0.5, torch.full_like(slip, params.r + params.h),
                                 params.r + params.h * (slip - 0.15) / 0.35))
    theta11 = _safe_acos(_safe_div(params.r * torch.cos(theta1), rj.clamp_min(params.eps), params.eps), params.acos_eps)
    jx_pos = rs * ((theta11 - thetam) - (1.0 - slip) * (torch.sin(theta11) - torch.sin(thetam)))
    jx_neg = -rs * ((theta11 - thetam) - _safe_div(torch.sin(theta11) - torch.sin(thetam), 1.0 + 0.9 * slip, params.eps))
    tan_abs_beta = _safe_tan(torch.abs(beta), params.max_tan_angle)
    jy_pos = params.r * (1.0 - slip) * (theta11 - thetam) * tan_abs_beta
    jy_neg = _safe_div(params.r * (theta11 - thetam) * tan_abs_beta, 1.0 + 0.99 * slip, params.eps)
    jx = torch.where(slip >= 0.0, jx_pos, jx_neg)
    jy = torch.where(slip >= 0.0, jy_pos, jy_neg)

    exp_x = 1.0 - _safe_exp_neg(_safe_div(jx, shear_k, params.eps), params.max_exp_arg)
    exp_y = 1.0 - _safe_exp_neg(_safe_div(jy, shear_k, params.eps), params.max_exp_arg)
    tan_phi = _safe_tan(phi, params.max_tan_angle)
    tau_m = (cohesion + sigma_m * tan_phi) * exp_x
    fs = params.r * params.b * cc * (cohesion + sigma_m * _safe_tan(0.5 * phi, params.max_tan_angle)) * exp_y
    fn = (params.r * params.b * aa * sigma_m + rs * params.b * tau_m * bb).clamp(0.0, params.fn_max)

    coef_t = 1.0 + params.c_T1 * ((params.fn_ave - fn) / params.fn_ave)
    r_aa = _safe_den(params.r * aa, params.eps)
    mr1 = rs * rs * cc * (params.b * cohesion + _safe_div(coef_t * fn * tan_phi, r_aa, params.eps)) * exp_x
    mr2 = 1.0 + _safe_div(rs * bb * tan_phi * exp_x, r_aa, params.eps)
    mr = _safe_div(mr1, mr2, params.eps)
    fdp = ((_safe_div(aa, cc, params.eps) + _safe_div(bb * bb, aa * cc, params.eps)) * mr / rs - _safe_div(bb * fn, aa, params.eps))
    fdp = fdp * ((1.0 + params.c_d1 + params.c_d2 * slip) * (1.0 + params.c_d3 * ((params.fn_ave - fn) / params.fn_ave)))

    fx = torch.where(slip >= 0.0, fdp, -fdp)
    fx = torch.where((slip < 0.0) & (fx > 0.0), torch.zeros_like(fx), fx)
    fx = torch.where(slip < 0.0, fx - 200.0 * torch.abs(vx), torch.maximum(fx, torch.full_like(fx, -100.0)))
    fy = torch.where(torch.abs(vy) > 1e-6, -torch.sign(vy) * torch.abs(fs), torch.zeros_like(vy))
    fy = fy - (600.0 * vy).clamp(-0.25 * fn, 0.25 * fn)
    dz = torch.where(vz > 0.0, torch.full_like(vz, params.first_damp_coef), torch.full_like(vz, params.second_damp_coef))
    fz_sub = (dz * vz).clamp(-dz * 5.0, dz * 5.0)
    fx = fx.clamp(-0.8 * fn, 0.8 * fn)
    fy = fy.clamp(-0.3 * fn, 0.3 * fn)
    fz = torch.maximum(torch.zeros_like(fn), fn - fz_sub).clamp(0.0, params.fn_max)
    fx = torch.where(omega < 0.0, -fx, fx)
    fx = torch.where(omega * vx < 0.0, fx - 200.0 * vx, fx)

    force = _finite(torch.stack([fx, fy, fz], dim=-1), limit=params.fn_max * 2.0)
    contact = sinkage > params.contact_threshold
    if in_contact is not None:
        contact = contact & (in_contact > 0.5)
    force = torch.where(contact.unsqueeze(-1), force, torch.zeros_like(force))
    return {"force": force, "slip": slip, "beta": beta, "sinkage": sinkage, "sigma_m": sigma_m}
