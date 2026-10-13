"""Manually invoked final pipeline report and honest V5 comparison."""
import argparse
import json
from pathlib import Path
import shutil


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run_dir')
    p.add_argument('--publish_best')
    p.add_argument('--v5_report')
    p.add_argument('--register_checkpoint')
    p.add_argument('--stage',choices=['state_teacher','state_student','force','physics','joint'])
    p.add_argument('--stage_dir')
    p.add_argument('--expected_ablation',choices=['v6_a','v6_b','v6_c','v6_d'])
    p.add_argument('--config_path')
    p.add_argument('--needs_physics_stage',action='store_true')
    args=p.parse_args()
    if args.needs_physics_stage:
        if not args.config_path: p.error('--config_path required')
        config=json.loads(Path(args.config_path).read_text())
        print('yes' if config['enable_dynamics_update'] else 'no')
        return
    if args.register_checkpoint:
        if not (args.stage and args.stage_dir and args.expected_ablation): p.error('registration requires stage, stage_dir and expected_ablation')
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        from models.checkpoint_v6 import read_checkpoint,restore_model
        ckpt=read_checkpoint(args.register_checkpoint)
        if ckpt.get('version')!=6 or ckpt.get('train_stage')!=args.stage or ckpt['v6_config']['ablation']!=args.expected_ablation:
            raise ValueError('explicit skipped-stage checkpoint version/stage/ablation mismatch')
        if args.config_path:
            expected=json.loads(Path(args.config_path).read_text())
            actual=json.loads(json.dumps(ckpt['v6_config']))
            if expected!=actual: raise ValueError('skipped checkpoint module/physics configuration differs from current pipeline')
        restore_model(ckpt)
        directory=Path(args.stage_dir)
        directory.mkdir(parents=True,exist_ok=False)
        shutil.copyfile(args.register_checkpoint,directory/'best.pt')
        status={'stage':args.stage,'status':'skipped_with_explicit_checkpoint','source_checkpoint':str(Path(args.register_checkpoint).resolve()),
            'best_checkpoint':str((directory/'best.pt').resolve()),'best_validation_score':ckpt['best_metric']}
        (directory/'stage_status.json').write_text(json.dumps(status,indent=2)+'\n')
        return
    if not (args.run_dir and args.publish_best): p.error('--run_dir and --publish_best required for final report')
    root=Path(args.run_dir)
    statuses={}
    for stage in ('state_teacher','state_student','force','physics','joint'):
        file=root/stage/'stage_status.json'
        if file.exists():
            status=json.loads(file.read_text())
            if status['status'] not in ('complete','skipped_with_explicit_checkpoint') or not Path(status['best_checkpoint']).is_file(): raise ValueError(f'incomplete stage {stage}')
            statuses[stage]=status
        elif stage=='physics': statuses[stage]={'status':'skipped: not applicable to V6-A/B'}
        else: raise FileNotFoundError(file)
    val=json.loads((root/'validation_report.json').read_text())
    test=json.loads((root/'test_report.json').read_text())
    final=Path(statuses['joint']['best_checkpoint'])
    if Path(val['checkpoint'])!=final or Path(test['checkpoint'])!=final: raise ValueError('evaluation checkpoint does not match current Joint stage')
    comparison={'status':'待对比：没有同划分、同评价协议的 V5 报告','v5':None}
    if args.v5_report:
        v5=json.loads(Path(args.v5_report).read_text())
        if v5.get('split_manifest')==test.get('split_manifest') and v5.get('split')=='test' and v5.get('metrics') and v5.get('evaluation_protocol')==test.get('evaluation_protocol') and test.get('evaluation_protocol'):
            comparison={'status':'same split and metric protocol','v5':v5['metrics'],'v6':test['metrics']}
        else: comparison['status']='待对比：V5 报告缺少一致的 split_manifest 或评价协议'
    summary={'stage_status':statuses,'final_best_checkpoint':str(final),'validation':val,'test':test,'v5_comparison':comparison,
             'continuous_lf_conditioned':str(root/'lf_conditioned/metrics_v6.json'),
             'autonomous':str(root/'autonomous/metrics_v6.json')}
    target=Path(args.publish_best);target.parent.mkdir(parents=True,exist_ok=True)
    # Publish only after current stages and both evaluations complete successfully.
    temp=target.with_suffix('.pt.tmp');shutil.copyfile(final,temp);temp.replace(target)
    out=root/'final_report.json';out.write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    (target.parent/'latest_run.txt').write_text(str(root.resolve())+'\n')
    print(json.dumps(statuses,indent=2,ensure_ascii=False))
    print(f'Final best checkpoint: {target}\nValidation report: {root / "validation_report.json"}\nFinal report: {out}')
    for name in ('body/position','body/velocity','body/attitude_rad','wheel0/force_Fx','wheel0/force_Fy','wheel0/force_Fz'):
        if name in test['metrics']:
            m=test['metrics'][name]
            print(f'{name}: MAE={m["MAE"]:.6g} RMSE={m["RMSE"]:.6g} P95={m["P95"]:.6g} Improvement Ratio={m["Improvement_Ratio"]} Harmful Ratio={m["Harmful_Ratio"]:.6g}')
    print(comparison['status'])


if __name__=='__main__': main()
