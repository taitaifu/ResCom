"""Manual acceptance suite. Never imported by production entry points."""
import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class PhysicsTests(unittest.TestCase):
    def test_deferred_metrics_equal_eager(self):
        import torch
        from models.loss_utils_v6 import PhysicalMetrics
        for device in (['cpu','cuda'] if torch.cuda.is_available() else ['cpu']):
            eager=PhysicalMetrics();deferred=PhysicalMetrics(defer_transfer=True)
            pred=torch.tensor([[1.,2.],[9.,8.],[float('nan'),3.]],device=device)
            truth=torch.zeros_like(pred);lf=torch.full_like(pred,4.)
            mask=torch.tensor([[True],[False],[True]],device=device)
            raw=torch.tensor([1.,-2.],device=device);gate=torch.tensor([.25,.5],device=device)
            for metric in (eager,deferred):
                metric.add('body/position',pred,truth,lf,mask,scale=pred.new_tensor(2.))
                metric.gate('body/position',raw,gate)
                metric.gate('body/position',raw,gate,raw*.1)
            self.assertTrue(deferred.pending)
            self.assertEqual(eager.quick_mae('body/position'),deferred.quick_mae('body/position'))
            self.assertFalse(deferred.pending)
            for metric in (eager,deferred): metric.add('body/position',pred[:1],truth[:1],lf[:1],scale=2.)
            self.assertEqual(eager.summarize(),deferred.summarize())
            self.assertFalse(deferred.pending)

    def test_device_scaler_cache_is_reused_and_invalidated(self):
        import numpy as np
        import torch
        from models.data_utils_v6 import GroupStandardizer,NumpyStandardScaler
        from models.runtime_v6 import scaler_values
        scaler=GroupStandardizer();stats=NumpyStandardScaler().fit(np.array([[1.,2.],[3.,4.]],np.float32))
        scaler.scalers['example']=stats
        first=scaler_values(scaler,'example',torch.device('cpu'))
        second=scaler_values(scaler,'example',torch.device('cpu'))
        self.assertTrue(all(a is b for a,b in zip(first,second)))
        stats.mean_=stats.mean_+1.
        third=scaler_values(scaler,'example',torch.device('cpu'))
        self.assertTrue(torch.equal(third[0],first[0]+1.))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'scaler.joblib';scaler.save(str(path))
            restored=GroupStandardizer.load(str(path))
            self.assertFalse(hasattr(restored,'_v6_device_statistics'))

    def test_running_mae_and_packed_loss_log(self):
        import torch
        from models.loss_utils_v6 import PhysicalMetrics,LossLedger
        metrics=PhysicalMetrics()
        metrics.add('body/position',torch.tensor([1.,3.]),torch.zeros(2),torch.ones(2))
        metrics.add('body/position',torch.tensor([5.]),torch.zeros(1),torch.ones(1))
        self.assertEqual(metrics.quick_mae('body/position'),3.)
        self.assertEqual(metrics.quick_mae('missing'),0.)
        reference=torch.tensor(2.,requires_grad=True)
        ledger=LossLedger(reference)
        ledger.add('active',reference.square(),.5)
        ledger.add('disabled',reference,1.,False)
        report=ledger.scalars()
        self.assertEqual(report['loss/active/raw'],4.)
        self.assertEqual(report['loss/active/weighted'],2.)
        self.assertFalse(report['loss/disabled/enabled'])
        self.assertEqual(report['total'],2.)

    def test_vectorized_quaternions_and_column_merge(self):
        import numpy as np
        import pandas as pd
        from models.data_utils_v6 import _quat_to_matrix_np,_quat_batch_to_matrix_np,_merge_columns
        q=np.array([[1.,0.,0.,0.],[0.,0.,0.,0.],[.2,.3,-.4,.5]])
        expected=np.stack([_quat_to_matrix_np(item) for item in q])
        self.assertTrue(np.allclose(_quat_batch_to_matrix_np(q),expected,rtol=1e-12,atol=1e-12))
        frame=pd.DataFrame({'lf_omega':[2.,3.],'valid':[True,False]})
        merged=_merge_columns(frame,{'lf_omega':np.array([4.,5.]),'new_valid':np.array([False,True])})
        self.assertTrue(merged.columns.is_unique)
        self.assertTrue(np.array_equal(merged.valid,frame.valid))
        self.assertTrue(np.array_equal(merged.lf_omega,[4.,5.]))
        self.assertTrue(np.array_equal(frame.lf_omega,[2.,3.]))

    def test_safe_loader_defaults(self):
        from train.train_graph_model_v6 import parse_args,make_loader
        args=parse_args(['--device','cpu'])
        self.assertEqual(args.num_workers,0)
        self.assertEqual(args.gradient_log_interval,100)
        loader=make_loader([{'body':0}],args)
        self.assertEqual(loader.num_workers,0)
        self.assertIsNone(loader.prefetch_factor)

    def test_workers_prepare_cache_before_iteration(self):
        import multiprocessing
        import torch
        from train.train_graph_model_v6 import parse_args,make_loader
        if 'fork' not in multiprocessing.get_all_start_methods():
            self.skipTest('in-memory V6 worker sharing requires fork')
        class ProbeDataset(torch.utils.data.Dataset):
            ready=False
            def prepare_worker_cache(self): self.ready=True
            def __len__(self): return 1
            def __getitem__(self,index): return {'body':torch.zeros(1)}
        dataset=ProbeDataset()
        args=parse_args(['--device','cpu','--num_workers','2'])
        previous=torch.multiprocessing.get_sharing_strategy()
        try:
            loader=make_loader(dataset,args)
            self.assertTrue(dataset.ready)
            self.assertEqual(loader.prefetch_factor,1)
            self.assertEqual(loader.multiprocessing_context.get_start_method(),'fork')
            self.assertEqual(torch.multiprocessing.get_sharing_strategy(),'file_system')
        finally: torch.multiprocessing.set_sharing_strategy(previous)

    def test_so3_zero_pi_and_quaternion_roundtrip(self):
        import torch
        from models.physics_modules_v6 import so3_exp,so3_log,matrix_quaternion,quat_matrix
        v=torch.tensor([[0.,0.,0.],[.1,.2,.3],[3.1415,0.,0.],[-3.1415,0.,0.]],requires_grad=True)
        r=so3_exp(v)
        self.assertTrue(torch.allclose(r,quat_matrix(matrix_quaternion(r)),atol=1e-5))
        self.assertTrue(torch.allclose(r,so3_exp(so3_log(r)),atol=1e-4))
        so3_log(r).square().mean().backward()
        self.assertTrue(torch.isfinite(v.grad).all())
    def test_independent_module_switches(self):
        switches = [
            {'enable_physics_edge': True}, {'enable_physics_attention': True},
            {'enable_contact_history': True}, {'enable_contact_edge_decoder': True},
            {'enable_force_gate': True}, {'enable_force_state_coupling': True},
            {'enable_force_state_coupling': True, 'enable_dynamics_update': True},
            {'enable_force_state_coupling': True, 'enable_dynamics_update': True,
             'enable_physics_refinement': True, 'physics_iterations': 2},
        ]
        for overrides in switches: smoke('v6_a','cpu',overrides)

    def test_sequence_loss_masks_and_pushforward(self):
        import numpy as np
        import pandas as pd
        import torch
        from train.train_graph_model_v6 import parse_args, compute_batch
        from models.config_v6 import config_from_args
        from models.checkpoint_v6 import build_model
        from models.data_utils_v6 import (load_column_spec, normalize_optional_columns,
            GroupStandardizer, GraphTemporalSequenceDatasetV6, graph_temporal_collate_fn,ROCKER_NAMES)
        from models.loss_utils_v6 import PhysicalMetrics
        spec=load_column_spec(str(ROOT/'Feature_Selection/DataSet'))
        cols=set(spec.base_feature_cols+spec.proxy_feature_cols+spec.res_cols+spec.target_cols)
        for source in ('lf','hf'):
            for name in ROCKER_NAMES:
                cols.update(f'{source}_susp_rocker_{name}_{s}' for s in ('pos_x','pos_y','pos_z','vel_x','vel_y','vel_z','q0','q1','q2','q3'))
        df=pd.DataFrame({c:np.zeros(40,np.float32) for c in cols})
        df['case_name']='case0';df['time']=np.arange(40)*.1
        for c in list(df):
            if c.endswith('_q0'): df[c]=1.
            if c.endswith('_pos_x'): df[c]=np.arange(40)*.01
            if c.endswith('_vel_x'): df[c]=.1
            if c.endswith('_Fz'): df[c]=10.
        for i in range(6): df[f'lf_wheel{i}_omega']=2.;df[f'lf_wheel{i}_in_contact']=1.;df[f'lf_wheel{i}_sinkage']=.01
        geometry=json.loads((ROOT/'vehicle_config/Zhurong_Config.json').read_text())
        for source in ('lf','hf'):
            for i,name in enumerate(('lf','rf','lm','rm','lb','rb')):
                for axis,value in zip('xyz',geometry['wheel_rel_pos'][name]):
                    df[f'{source}_wheel{i}_pos_{axis}']=value+(np.arange(40)*.01 if axis=='x' else 0.)
            for name in ROCKER_NAMES:
                for axis,value in zip('xyz',geometry['rocker_pos'][name]):
                    df[f'{source}_susp_rocker_{name}_pos_{axis}']=value+(np.arange(40)*.01 if axis=='x' else 0.)
        for c in df:
            if c.startswith('hf_') and c.endswith('_pos_x'): df[c]+=.02
        for c in spec.res_cols:
            h='hf_'+c[4:];l='lf_'+c[4:]
            if h in df and l in df: df[c]=df[h]-df[l]
        df=normalize_optional_columns(df,spec)
        scaler=GroupStandardizer().fit(df,spec)
        for preset in ('v6_a','v6_d'):
            args=parse_args(['--ablation',preset,'--device','cpu','--body_rollout_len','2',
                '--hidden_dim','16','--tcn_dim','24','--lstm_dim','16','--lstm_layers','1',
                '--force_hidden_dim','8','--num_workers','0'])
            cfg=config_from_args(args)
            ds=GraphTemporalSequenceDatasetV6(df,spec,scaler,history_len=9,teacher_future_len=3,body_rollout_len=2);ds.config=cfg
            batch=graph_temporal_collate_fn([ds[0],ds[1]])
            legacy=ds[0]
            ds.compact_training_samples=True
            compact=ds[0]
            for key,value in compact.items():
                self.assertIn(key,legacy)
                self.assertTrue(torch.equal(value,legacy[key]),key)
            self.assertNotIn('hf_body_pose',compact)
            self.assertIn('v6_hf_body',compact)  # labels remain for loss only
            ds.compact_training_samples=False
            if preset=='v6_a':
                self.assertTrue(all(np.count_nonzero(history)==0 for history in ds.v6_shear.values()))
                self.assertEqual(ds.v6_physical.strides[0],0)
                self.assertTrue(all(history.strides[0]==0 for history in ds.v6_shear.values()))
            cached=ds.v6_raw
            ds.prepare_worker_cache()
            self.assertIs(ds.v6_raw,cached)
            self.assertFalse(ds.v6_raw['lf_body'].flags.writeable)
            model=build_model(spec,args,cfg);model.omega_supervised=[False]*6
            params=np.zeros((6,7),np.float32);params[:,0]=10.
            ledger=compute_batch(model,batch,'joint',None,params,spec,scaler,cfg,args,PhysicalMetrics())
            self.assertEqual(ledger.entries['force/dyn'][2],False)
            self.assertIn('state/wheel_local_reg',ledger.entries)
            self.assertGreater(float(ledger.entries['state/kin'][0]),0.)
            has_push=any(k.startswith('pushforward/') for k in ledger.entries)
            self.assertEqual(has_push,cfg.enable_pushforward)
            ledger.total.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            if preset=='v6_a':
                cfg.dynamics_reference_is_vehicle_com=True
                cfg.dynamics_calibrated=True;cfg.inertia_body=(2.,3.,4.);cfg.validate()
                args.lambda_dyn=.001
                calibrated=compute_batch(model,batch,'joint',None,params,spec,scaler,cfg,args,PhysicalMetrics())
                dyn=calibrated.entries['force/dyn'][0]
                self.assertGreater(float(dyn),0.)
                dg=torch.autograd.grad(dyn,model.force_residual_head.net[-1].weight,retain_graph=True)[0]
                self.assertGreater(float(dg.abs().sum()),0.)
                lr=calibrated.entries['state/wheel_local_reg'][0]
                self.assertGreater(float(lr),0.)
            from models.checkpoint_v6 import restore_model
            from train.train_graph_model_v6 import payload
            split={'train':['case0'],'val':[],'test':[]}
            opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
            opt.step()
            ckpt=payload(model,spec,scaler,args,cfg,params,split,1,1.,opt)
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'complete.pt';torch.save(ckpt,path)
                restored,rspec,rscaler,rcfg=restore_model(torch.load(path,weights_only=False))
                self.assertEqual(rspec.to_dict(),spec.to_dict())
                self.assertEqual(rcfg.to_dict(),cfg.to_dict())
                for key,value in model.state_dict().items(): self.assertTrue(torch.equal(value,restored.state_dict()[key]))
                resumed_optimizer=torch.optim.AdamW(restored.parameters(),lr=1e-4)
                resumed_optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                self.assertEqual(ckpt['resume_format'],'v6_complete_epoch_v1')
                self.assertEqual(len(resumed_optimizer.state),len(opt.state))

    def test_contact_frames_and_aggregation(self):
        import torch
        from models.physics_modules_v6 import contact_frame, aggregate_contact
        n = torch.tensor([0., 0., 1.]).expand(2, 1, 6, 3)
        h = torch.tensor([1., 0., 0.]).expand_as(n)
        r = contact_frame(n, h)
        self.assertTrue(torch.allclose(r.transpose(-1, -2) @ r, torch.eye(3).expand_as(r)))
        f = torch.zeros_like(n); f[..., 2] = 10
        p = torch.zeros_like(n); p[..., 0] = 1
        force, torque, world = aggregate_contact(f, r, p, torch.zeros(2, 1, 3))
        self.assertTrue(torch.allclose(force[..., 2], torch.full((2, 1), 60.)))
        self.assertTrue(torch.allclose(torque[..., 1], torch.full((2, 1), -60.)))
        slope = torch.tensor([0., .6, .8]).expand_as(n)
        sr = contact_frame(slope, h)
        self.assertTrue(torch.allclose(sr[..., :, 2], slope))
        self.assertTrue(torch.allclose(world, f))

    def test_kinematics_is_not_an_identity(self):
        import torch
        from models.loss_utils_v6 import independent_kinematics
        p = torch.tensor([[[0., 0., 0.], [1., 0., 0.]]], requires_grad=True)
        v = torch.zeros_like(p, requires_grad=True)
        loss = independent_kinematics(p, v, torch.ones(1, 1))
        self.assertGreater(float(loss), 0)
        loss.backward()
        self.assertGreater(float(v.grad.abs().sum()), 0)

    def test_gate_direction_and_gradient(self):
        import torch
        from models.loss_utils_v6 import correction_gate_loss
        raw = torch.tensor([[[1., -1., .001]]], requires_grad=True)
        gate = torch.full_like(raw, .1, requires_grad=True)
        target = torch.ones_like(raw)
        loss = correction_gate_loss(raw, gate, target, torch.ones_like(raw))
        loss.backward()
        self.assertTrue(torch.isfinite(gate.grad).all())
        self.assertLess(float(gate.grad[0, 0, 0]), 0)
        self.assertGreater(float(gate.grad[0, 0, 1]), 0)

    def test_config_dependencies_and_overrides(self):
        from models.config_v6 import V6Config
        for name in ('v6_a', 'v6_b', 'v6_c', 'v6_d'):
            V6Config.preset(name).validate()
        c = V6Config.preset('v6_a')
        self.assertFalse(any(getattr(c, k) for k in c.switch_names()))
        c.enable_dynamics_update = True
        with self.assertRaises(ValueError): c.validate()

    def test_v5_sources_preserved(self):
        manifest = json.loads((ROOT / 'v6_tests/v5_source_hashes.json').read_text())
        for name, digest in manifest.items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest, name)

    def test_optional_hf_omega_keeps_lf(self):
        import numpy as np
        import pandas as pd
        from models.data_utils_v6 import normalize_optional_columns, load_column_spec
        spec = load_column_spec(str(ROOT / 'Feature_Selection/DataSet'))
        df = pd.DataFrame({'case_name': ['case0'] * 3, 'time': [0., .1, .2],
                           'lf_wheel0_omega': [2., 3., 4.]})
        out = normalize_optional_columns(df, spec, require_labels=False)
        self.assertTrue(np.allclose(out.lf_wheel0_omega, [2., 3., 4.]))
        self.assertFalse(out.v6_hf_wheel0_omega_valid.any())
        self.assertTrue(out.v6_lf_wheel0_omega_valid.all())


def smoke(ablation, device, overrides=None):
    """Real minimal forward/backward, optimizer and checkpoint roundtrip."""
    import torch
    from models.config_v6 import V6Config
    from models.graph_temporal_hgt_compensation_v6 import HGTGraphTemporalCompensationModelV6
    from models.data_utils_v6 import ROCKER_NAMES
    cfg = V6Config.preset(ablation)
    for k,v in (overrides or {}).items(): setattr(cfg,k,v)
    cfg.validate()
    window=cfg.history_len+1
    dims = {'system': 3, 'body': 6, 'res_body': 6}
    dims.update({n: 7 for n in ROCKER_NAMES})
    for i in range(6): dims.update({f'wheel{i}_kin': 7, f'wheel{i}_contact': 7, f'res_wheel{i}_kin': 3})
    model = HGTGraphTemporalCompensationModelV6(dims, node_hidden_dim=16, tcn_hidden_dim=24,
        lstm_hidden_dim=16, lstm_layers=1, force_hidden_dim=8, dropout=0., config=cfg).to(device)
    if cfg.enable_contact_edge_decoder:
        with torch.no_grad(): model.contact_edge_decoder.net[-1].bias.fill_(.01)
    batch = {k: torch.randn(2, window, d, device=device) for k, d in dims.items() if not k.startswith('res_')}
    batch['physics_nodes'] = torch.randn(2, window, model.num_nodes, 32, device=device)
    out = model.forward_state(batch)
    ref = out['z']
    zero = ref.new_zeros(2, 1, 3)
    rot = torch.eye(3, device=device).expand(2, 1, 3, 3)
    state = dict(z=ref, body_node=out['body_node'], wheel_kin_nodes=out['wheel_kin_nodes'],
        wheel_contact_nodes=out['wheel_contact_nodes'], body_position=zero,
        body_velocity=zero, body_acceleration=zero, body_rotation=rot, body_angular_velocity=zero,
        wheel_position={i: zero + i*.1 for i in range(6)}, wheel_velocity={i: zero for i in range(6)},
        wheel_rotation={i: rot for i in range(6)}, wheel_omega={i: ref.new_zeros(2, 1, 1) for i in range(6)},
        force_base={i: zero + 1 for i in range(6)}, fphy_z={i: ref.new_ones(2, 1, 1) for i in range(6)},
        rockers={}, lf_force={i: zero for i in range(6)})
    for i in range(6):
        batch[f'force_history_wheel{i}'] = torch.randn(2, window, 7, device=device)
        batch[f'shear_history_wheel{i}'] = torch.randn(2, window, 8, device=device)
        c = torch.zeros(2, 1, 15, device=device); c[..., 0] = 1; c[..., 3] = 1; c[..., 4] = .01
        batch[f'contact_raw_wheel{i}'] = c
    touched = {}
    hooks = []
    for name in ('contact_edge_decoder', 'force_gate_head', 'enhanced_history', 'dynamics_head'):
        module = getattr(model, name, None)
        if module is not None:
            def count(_m, _i, _o, name=name): touched[name] = touched.get(name, 0) + 1
            hooks.append(module.register_forward_hook(count))
    final, force = model.coupled_update(state, batch)
    loss = sum(v.square().mean() for k, v in out.items() if isinstance(v, torch.Tensor) and k.startswith('pred_'))
    loss = loss + sum(force[f'force_pred_wheel{i}'].square().mean() for i in range(6)) + final['body_velocity'].square().mean()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad(); loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    if cfg.enable_force_gate:
        assert model.force_gate_head[-1].weight.grad is not None
        assert model.force_gate_head[-1].weight.grad.abs().sum()>0
    if cfg.enable_dynamics_update:
        assert model.dynamics_head.net[-1].weight.grad is not None
    assert model.body_pos_gate_head.net[-1].weight.grad is None or torch.isfinite(model.body_pos_gate_head.net[-1].weight.grad).all()
    optimizer.step()
    assert bool(touched.get('contact_edge_decoder')) == cfg.enable_contact_edge_decoder
    assert bool(touched.get('force_gate_head')) == cfg.enable_force_gate
    assert bool(touched.get('enhanced_history')) == cfg.enable_contact_history
    assert bool(touched.get('dynamics_head')) == cfg.enable_dynamics_update
    no_contact = dict(batch)
    for i in range(6):
        no_contact[f'contact_raw_wheel{i}'] = batch[f'contact_raw_wheel{i}'].clone()
        no_contact[f'contact_raw_wheel{i}'][..., 0] = 0
    _, empty_force = model.coupled_update(state, no_contact)
    for i in range(6): assert torch.count_nonzero(empty_force[f'force_pred_wheel{i}']) == 0
    from models.physics_modules_v6 import contact_features
    feats,_,_,_=contact_features(state,batch,cfg)
    for i in range(6):
        assert torch.allclose(feats[i][...,10:12],torch.zeros_like(feats[i][...,10:12]))
    # Steering changes axle/terrain frame while preserving orthonormality.
    from models.physics_modules_v6 import so3_exp
    turn=dict(state);turn['wheel_rotation']={i:so3_exp(zero+zero.new_tensor([0.,0.,.6])) for i in range(6)}
    _,turn_force=model.coupled_update(turn,batch)
    for i in range(6):
        frame=turn_force[f'contact_frame_wheel{i}']
        assert torch.allclose(frame.transpose(-1,-2)@frame,torch.eye(3,device=device).expand_as(frame),atol=1e-5)
    stripped = dict(batch); stripped['hf_body'] = torch.randn(2, 1, 6, device=device)*1e6
    model.eval()
    with torch.no_grad():
        a, b = model.forward_state(batch), model.forward_state(stripped)
    def compare_state_outputs(left, right, prefix='state', exact=False):
        assert left.keys() == right.keys(), f'{prefix}: output keys differ'
        for key, value in left.items():
            path = f'{prefix}/{key}'
            other = right[key]
            if isinstance(value, dict):
                compare_state_outputs(value, other, path, exact)
            elif isinstance(value, torch.Tensor):
                assert torch.isfinite(value).all(), f'{path}: nonfinite LF-only output'
                assert torch.isfinite(other).all(), f'{path}: nonfinite output with HF labels'
                # CUDA scatter_add_ uses floating-point atomic reductions; two
                # otherwise identical calls need not produce identical bits.
                torch.testing.assert_close(value, other, rtol=0. if exact else 1e-5,
                    atol=0. if exact else 1e-6, equal_nan=False,
                    msg=lambda message: f'{path}: HF-isolation comparison failed\n{message}')
    compare_state_outputs(a, b, exact=ref.device.type == 'cpu')
    if ref.device.type == 'cuda':
        # Retain an exact isolation oracle on CPU rather than relying only on
        # the CUDA tolerance. Run the same updated weights and input values.
        import copy
        cpu_model = copy.deepcopy(model).cpu().eval()
        cpu_batch = {key: value.detach().cpu() for key, value in batch.items()}
        cpu_stripped = dict(cpu_batch)
        cpu_stripped['hf_body'] = stripped['hf_body'].detach().cpu()
        with torch.no_grad():
            ca = cpu_model.forward_state(cpu_batch)
            cb = cpu_model.forward_state(cpu_stripped)
        compare_state_outputs(ca, cb, exact=True)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)/'roundtrip.pt'
        torch.save({'model_state_dict': model.state_dict(), 'v6_config': cfg.to_dict()}, path)
        saved = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(saved['model_state_dict'], strict=True)
    for h in hooks: h.remove()
    print(f'{ablation} {device}: forward/backward/optimizer/checkpoint/bypass/no-contact/HF-isolation completed')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--ablation', choices=['v6_a', 'v6_b', 'v6_c', 'v6_d', 'all'], default='all')
    p.add_argument('--device', default='cpu')
    p.add_argument('--config_json')
    args = p.parse_args()
    if args.smoke:
        overrides=json.loads(Path(args.config_json).read_text()) if args.config_json else None
        for name in (['v6_a','v6_b','v6_c','v6_d'] if args.ablation == 'all' else [args.ablation]): smoke(name, args.device,overrides)
    else:
        unittest.main(argv=[sys.argv[0]])
