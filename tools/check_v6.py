"""Manual-only preflight, syntax and optional full acceptance checks."""
import argparse
import ast
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--syntax',action='store_true')
    p.add_argument('--environment',action='store_true')
    p.add_argument('--feature_dir',default=str(ROOT/'Feature_Selection/DataSet'))
    p.add_argument('--merged_csv',default=str(ROOT/'Feature_Selection/DataSet/merged_error_dataset.csv'))
    p.add_argument('--full',action='store_true')
    p.add_argument('--data_smoke',action='store_true',help='manual real-data CPU minimal training, one train/val batch')
    args=p.parse_args()
    if args.syntax or args.full:
        paths=[*ROOT.glob('models/*v6.py'),ROOT/'train/train_graph_model_v6.py',ROOT/'infer/infer_graph_model_v6.py',
            ROOT/'tools/check_v6.py',ROOT/'tools/v6_report.py',ROOT/'v6_tests/test_v6.py']
        for path in paths: ast.parse(path.read_text(),filename=str(path))
        print(f'Parsed {len(paths)} Python files')
    if args.environment:
        import torch,numpy,pandas,joblib,tqdm
        files=[Path(args.merged_csv)]+[Path(args.feature_dir)/name for name in ('base_feature_columns.csv','proxy_feature_columns.csv','target_columns.csv','res_columns.csv')]
        for file in files:
            if not file.is_file() or file.stat().st_size==0: raise FileNotFoundError(file)
        print('Environment/data files available; torch='+torch.__version__+' CUDA='+str(torch.cuda.is_available()))
    if args.full:
        import unittest
        from v6_tests.test_v6 import PhysicsTests,smoke
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(PhysicsTests)
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful(): raise SystemExit(1)
        for name in ('v6_a','v6_b','v6_c','v6_d'): smoke(name,'cpu')
        import torch
        if torch.cuda.is_available():
            for name in ('v6_a','v6_b','v6_c','v6_d'): smoke(name,'cuda')
        else: print('CUDA smoke skipped: CUDA unavailable')
    if args.data_smoke:
        from train.train_graph_model_v6 import main as train_main
        train_main(['--ablation','v6_a','--train_stage','state_teacher','--device','cpu',
            '--feature_dir',args.feature_dir,'--merged_csv',args.merged_csv,'--epochs','1',
            '--max_batches','1','--batch_size','2','--num_workers','0','--body_rollout_len','2',
            '--hidden_dim','16','--tcn_dim','24','--lstm_dim','16','--lstm_layers','1',
            '--force_hidden_dim','8','--save_dir',str(ROOT/'results_v6/manual_checks')])


if __name__=='__main__': main()
