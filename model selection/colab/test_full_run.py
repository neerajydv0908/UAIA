"""Offline checks for full coverage, A100 gating and the single-run supervisor."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from full_run import require_a100, run, training_arguments, verify_completed_training, verify_probe
from protocol import save_json, sha256


class FullRunTests(unittest.TestCase):
    def test_a100_is_required(self):
        with patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.is_bf16_supported', return_value=True):
            with patch('torch.cuda.get_device_name', return_value='NVIDIA L4'):
                with self.assertRaisesRegex(RuntimeError, 'A100 was requested'): require_a100()
            with patch('torch.cuda.get_device_name', return_value='NVIDIA A100-SXM4-40GB'):
                self.assertIn('A100', require_a100())

    def test_full_pass_command_has_no_subset_or_sweep(self):
        args = training_arguments(Path('/bundle'), 4646)
        self.assertEqual(args[args.index('--updates') + 1], '1162')
        self.assertEqual(args[args.index('--pool') + 1], 'all_clear')
        self.assertEqual(args[args.index('--epochs') + 1], '1')
        self.assertEqual(args[args.index('--minimum-epochs') + 1], '1')
        self.assertEqual(args[args.index('--checkpoint-every') + 1], '100')
        self.assertNotIn('--max-pairs', args)
        self.assertNotIn('--resume', args)

    def test_complete_coverage_and_best_exposure(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            ids = [str(i) for i in range(4646)]
            save_json(folder/'plan.json', {'training_pair_ids': ids, 'planned_pair_batches': [ids[i:i+4] for i in range(0, len(ids), 4)]})
            updates = [{'step': i, 'pairs': 4 if i < 1162 else 2} for i in range(1, 1163)]
            (folder/'updates.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in updates))
            monitors = [{'step': i, 'log_loss': i/1000} for i in list(range(100, 1163, 100)) + [1162]]
            result = {'status':'completed', 'completed_updates':1162, 'stopped_early':False, 'monitors':monitors, 'best':monitors[0]}
            save_json(folder/'result.json', result)
            for step in (100, 1162):
                checkpoint = folder/f'checkpoint_{step:04d}';checkpoint.mkdir()
                (checkpoint/'weights').write_text('fixture')
                save_json(checkpoint/'receipt.json', {'files':{'weights':sha256(checkpoint/'weights')}})
            self.assertEqual(verify_completed_training(folder, 4646)[1], 400)
            updates[-1]['pairs'] = 1
            (folder/'updates.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in updates))
            with self.assertRaisesRegex(ValueError, 'every planned pair'): verify_completed_training(folder, 4646)

    def test_probe_rejects_changed_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle=Path(tmp);save_json(bundle/'manifest.json', {})
            result={'status':'passed', 'exact_restart_weights_optimizer_random_loss':True,
                    'gpu':'NVIDIA A100', 'bundle_sha256':sha256(bundle/'manifest.json'), 'source_sha256':{'train.py':'old'}}
            with self.assertRaisesRegex(ValueError, 'sources differ'): verify_probe(result, bundle, {'train.py':'new'})

    def test_pipeline_trains_once_and_stops_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bundle=root/'bundle';bundle.mkdir();save_json(bundle/'manifest.json', {'files_sha256':{}})
            calls=[]
            def worker(script, arguments, destination, minutes):
                calls.append(script);destination.mkdir()
                if script=='probe.py':
                    sources={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}
                    save_json(destination/'result.json', {'status':'passed','exact_restart_weights_optimizer_random_loss':True,
                        'gpu':'NVIDIA A100','bundle_sha256':sha256(bundle/'manifest.json'),'source_sha256':sources})
                if script=='evaluate.py':
                    save_json(destination/'full_report.json', {'views':{'averaged':{'matches':252}},'development_screen':{'accepted':True}})
            with patch('full_run.require_a100', return_value='NVIDIA A100'), patch('full_run.check_data', return_value=({'model':'test','revision':'pinned'},4646)):
                with patch('full_run.bounded', side_effect=worker), patch('full_run.verify_completed_training', return_value=({'best':{'step':1162}},4646)):
                    run(bundle,root/'success',360)
                    self.assertEqual(calls,['probe.py','train.py','evaluate.py','evaluate.py'])
                    with self.assertRaises(FileExistsError): run(bundle,root/'success',360)
                with patch('full_run.bounded', side_effect=RuntimeError('probe failed')) as failed:
                    with self.assertRaisesRegex(RuntimeError, 'probe failed'): run(bundle,root/'failure',360)
                    self.assertEqual(failed.call_count,1)
            self.assertEqual(json.loads((root/'success/status.json').read_text())['status'],'completed')
            self.assertEqual(json.loads((root/'failure/status.json').read_text())['status'],'failed')


if __name__=='__main__':
    unittest.main()
