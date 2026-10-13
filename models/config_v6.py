"""One versioned configuration shared by model, training and inference."""
from dataclasses import dataclass, asdict, fields
import argparse
from pathlib import Path


@dataclass
class V6Config:
    ablation: str = 'v6_a'
    enable_physics_edge: bool = False
    enable_physics_attention: bool = False
    enable_contact_history: bool = False
    enable_contact_edge_decoder: bool = False
    enable_force_gate: bool = False
    enable_force_state_coupling: bool = False
    enable_dynamics_update: bool = False
    enable_physics_refinement: bool = False
    enable_pushforward: bool = False
    physics_iterations: int = 1
    rollout_steps: int = 3
    history_len: int = 9
    wheel_radius: float = .135
    terrain_file: str = str(Path(__file__).resolve().parent/'TerrainPlugins/zero_terrain_reader.txt')
    shear_rho: float = .95
    omega_scale: float = 10.
    velocity_scale: float = 1.
    angular_velocity_scale: float = 1.
    position_scale: float = .1
    force_scale: float = 400.
    torque_scale: float = 400.
    # Input/export force convention used by V5's Fx/Fy/Fz baseline.
    # Override to world only when the exporting simulator confirms that convention.
    input_force_frame: str = 'contact'
    spin_axis: tuple = (0., 1., 0.)
    mass: float = 240.
    gravity: tuple = (0., 0., -9.81)
    inertia_body: tuple = (0., 0., 0.)
    dynamics_calibrated: bool = False
    dynamics_reference_is_vehicle_com: bool = False
    schema_version: int = 6

    @classmethod
    def switch_names(cls):
        return [f.name for f in fields(cls) if f.name.startswith('enable_')]

    @classmethod
    def preset(cls, name):
        if name not in ('v6_a', 'v6_b', 'v6_c', 'v6_d'): raise ValueError(name)
        c = cls(ablation=name)
        if name != 'v6_a':
            c.enable_physics_edge = c.enable_contact_edge_decoder = True
            c.enable_contact_history = c.enable_force_gate = True
        if name in ('v6_c', 'v6_d'):
            c.enable_force_state_coupling = c.enable_dynamics_update = True
        if name == 'v6_d':
            c.enable_physics_attention = c.enable_physics_refinement = c.enable_pushforward = True
        return c

    def validate(self):
        if self.ablation not in ('v6_a','v6_b','v6_c','v6_d') or self.schema_version!=6:
            raise ValueError('unsupported V6 preset/schema')
        self.spin_axis=tuple(float(x) for x in self.spin_axis)
        self.gravity=tuple(float(x) for x in self.gravity)
        self.inertia_body=tuple(float(x) for x in self.inertia_body)
        if self.enable_dynamics_update and not self.enable_force_state_coupling:
            raise ValueError('dynamics_update requires force_state_coupling')
        if self.enable_physics_refinement and not (self.enable_force_state_coupling and self.enable_dynamics_update):
            raise ValueError('physics_refinement requires coupling and dynamics_update')
        if self.physics_iterations not in (1, 2): raise ValueError('physics_iterations must be 1 or 2')
        if self.physics_iterations == 2 and not self.enable_physics_refinement:
            raise ValueError('physics_iterations=2 requires physics_refinement')
        if not 2 <= self.rollout_steps <= 5: raise ValueError('rollout_steps must be 2..5')
        if self.history_len < 2: raise ValueError('history_len must be >=2')
        if self.input_force_frame not in ('contact', 'world'): raise ValueError('input_force_frame must be contact/world')
        if not 0 <= self.shear_rho <= 1: raise ValueError('shear_rho must be in [0,1]')
        for k in ('wheel_radius','omega_scale','velocity_scale','angular_velocity_scale','position_scale','force_scale','torque_scale','mass'):
            if getattr(self, k) <= 0: raise ValueError(f'{k} must be positive')
        if len(self.spin_axis) != 3 or sum(x*x for x in self.spin_axis) < 1e-10: raise ValueError('invalid spin_axis')
        if self.dynamics_calibrated and any(x <= 0 for x in self.inertia_body):
            raise ValueError('calibrated dynamics requires positive principal inertia')
        if self.dynamics_calibrated and not self.dynamics_reference_is_vehicle_com:
            raise ValueError('dyn_loss requires confirmed whole-vehicle COM state, not chassis origin')
        return self

    def to_dict(self): return asdict(self)

    @classmethod
    def from_dict(cls, d):
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown: raise ValueError(f'unknown V6 configuration: {sorted(unknown)}')
        return cls(**d).validate()


def add_config_arguments(parser):
    parser.add_argument('--ablation', choices=['v6_a','v6_b','v6_c','v6_d'], default='v6_a')
    for name in V6Config.switch_names():
        parser.add_argument('--'+name, action=argparse.BooleanOptionalAction, default=None)
    for name, typ in [('physics_iterations',int),('rollout_steps',int),('wheel_radius',float),('shear_rho',float),
                      ('force_scale',float),('torque_scale',float),('omega_scale',float),('mass',float)]:
        parser.add_argument('--'+name, type=typ, default=None)
    parser.add_argument('--input_force_frame', choices=['contact','world'], default=None)
    parser.add_argument('--terrain_file', default=None)
    parser.add_argument('--spin_axis', type=float, nargs=3, default=None)
    parser.add_argument('--inertia_body', type=float, nargs=3, default=None)
    parser.add_argument('--dynamics_calibrated', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--dynamics_reference_is_vehicle_com', action=argparse.BooleanOptionalAction, default=None)


def config_from_args(args):
    c = V6Config.preset(args.ablation)
    for f in fields(c):
        v = getattr(args, f.name, None)
        if v is not None: setattr(c, f.name, v)
    return c.validate()
