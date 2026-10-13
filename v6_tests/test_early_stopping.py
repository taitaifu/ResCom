"""Exercise V6 stage completion with controlled validation scores on CPU."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from train import train_graph_model_v6 as trainer


class EarlyStoppingTests(unittest.TestCase):
    def run_stage(self, scores, extra=(), split_at=None):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder) / 'teacher'
            model = torch.nn.Linear(1, 1)
            model.role = 'teacher'
            spec = trainer.load_column_spec(str(trainer.ROOT / 'Feature_Selection/DataSet'))
            scaler = SimpleNamespace(scalers={}, save=lambda path: None)
            train = {f'v6_hf_wheel{i}_omega_valid': np.array([False]) for i in range(6)}
            def build_model(spec, args, cfg, role):
                model.config = cfg
                return model
            def run_epoch(*args, **kwargs):
                epoch = kwargs.get('epoch', args[-1])
                return {'losses': {'total': 1.}, 'selection_score': scores[epoch - 1]}
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(trainer, 'data_bundle', return_value=(spec, scaler, train, [[0], [0], [0]], {})))
                stack.enter_context(patch.object(trainer, 'build_model', side_effect=build_model))
                stack.enter_context(patch.object(trainer, 'run_epoch', side_effect=run_epoch))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                options = ['--device', 'cpu', '--run_dir', str(directory)]
                trainer.main([*options, '--epochs', str(split_at or len(scores)), *extra])
                if split_at is not None:
                    trainer.main([*options, '--epochs', str(len(scores)), '--resume_ckpt', str(directory / 'last_epoch.pt'), *extra])
            status = json.loads((directory / 'stage_status.json').read_text())
            best = torch.load(directory / 'best.pt', weights_only=False)
            last = torch.load(directory / 'last_epoch.pt', weights_only=False)
            records = [json.loads(line) for line in (directory / 'epochs.jsonl').read_text().splitlines()]
            return status, best, last, records

    def test_default_stops_after_fifteen_stale_epochs_and_preserves_best(self):
        status, best, last, records = self.run_stage([1.] * 30)
        self.assertEqual(len(records), 16)
        self.assertEqual(best['epoch'], 1)
        self.assertEqual(last['epoch'], 16)
        self.assertEqual(last['early_stopping']['epochs_without_improvement'], 15)
        self.assertEqual(status['status'], 'complete')
        self.assertTrue(status['early_stopped'])

    def test_improvement_resets_counter(self):
        status, best, last, records = self.run_stage([3., 3., 2., 2., 2., 2.], ['--patience', '2'])
        self.assertEqual(len(records), 5)
        self.assertEqual(best['epoch'], 3)
        self.assertEqual(last['early_stopping']['best_epoch'], 3)
        self.assertTrue(status['early_stopped'])

    def test_epoch_limit_still_applies(self):
        status, best, last, records = self.run_stage([3., 2., 1.])
        self.assertEqual(len(records), 3)
        self.assertEqual(best['epoch'], 3)
        self.assertFalse(status['early_stopped'])

    def test_resume_preserves_stale_epoch_counter(self):
        status, best, last, records = self.run_stage([1.] * 30, split_at=10)
        self.assertEqual(len(records), 16)
        self.assertEqual(best['epoch'], 1)
        self.assertEqual(last['epoch'], 16)
        self.assertTrue(status['early_stopped'])

    def test_resuming_stopped_stage_does_not_train_extra_epoch(self):
        status, best, last, records = self.run_stage([1.] * 30, split_at=20)
        self.assertEqual(len(records), 16)
        self.assertEqual(last['epoch'], 16)
        self.assertEqual(status['epochs_completed'], 16)

    def test_nonpositive_patience_is_rejected(self):
        for value in ('0', '-1'):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    trainer.parse_args(['--patience', value])


if __name__ == '__main__':
    unittest.main()
