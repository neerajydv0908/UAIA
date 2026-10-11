"""One complete A100 training pass, then both-order validation. No automatic retries."""
import argparse
import json
import math
import shutil
import time
from pathlib import Path
from launch import bounded
from protocol import assert_separate, read_rows, save_json, sha256, verify_bundle


def require_a100():
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Choose a CUDA A100 runtime with BF16 support.')
    name = torch.cuda.get_device_name()
    if 'A100' not in name.upper():
        raise RuntimeError(f'An A100 was requested; assigned GPU is {name}. Select A100 in Colab before starting.')
    return name


def check_data(bundle):
    manifest = verify_bundle(bundle)
    training = read_rows(bundle / 'training.jsonl')
    monitor = read_rows(bundle / 'monitor.jsonl')
    clear = [r for r in training if r['target'] in ('A', 'B')]
    assert_separate(training, monitor, [{'id': g, 'group_id': g} for g in manifest['validation_groups']])
    if len(clear) != 4646 or len(monitor) != 400 or manifest['counts']['validation_pairs'] != 396:
        raise ValueError('This notebook requires the frozen 4,646-clear / 400-monitor / 396-validation bundle.')
    return manifest, len(clear)


def training_arguments(bundle, pairs):
    return ['--bundle', str(bundle), '--pool', 'all_clear', '--learning-rate', '2e-5',
            '--updates', str(math.ceil(pairs / 4)), '--epochs', '1', '--minimum-epochs', '1',
            '--warmup', '50', '--seed', '20261007', '--labels', 'hard', '--style', 'json',
            '--layers', '36', '--monitor-every', '100', '--checkpoint-every', '100', '--patience', '0']


def verify_probe(probe, bundle, sources):
    if probe['status'] != 'passed' or not probe['exact_restart_weights_optimizer_random_loss']:
        raise ValueError('The actual GPU readiness and exact restart checks must pass.')
    if 'A100' not in probe['gpu'].upper() or probe['bundle_sha256'] != sha256(bundle / 'manifest.json'):
        raise ValueError('The passing probe must use this A100 and this bundle.')
    if probe['source_sha256'] != sources:
        raise ValueError('Worker sources differ from the passing probe.')


def verify_completed_training(folder, pairs):
    result = json.loads((folder / 'result.json').read_text())
    plan = json.loads((folder / 'plan.json').read_text())
    updates = read_rows(folder / 'updates.jsonl')
    expected = math.ceil(pairs / 4)
    if result['status'] != 'completed' or result['completed_updates'] != expected or result['stopped_early']:
        raise ValueError('The full-data pass did not complete; no partial run can be called complete.')
    if [r['step'] for r in updates] != list(range(1, expected + 1)) or sum(r['pairs'] for r in updates) != pairs:
        raise ValueError('Training must cover every planned pair exactly once.')
    flattened = [i for batch in plan['planned_pair_batches'] for i in batch]
    if flattened != plan['training_pair_ids'] or len(set(flattened)) != pairs:
        raise ValueError('Saved training pair order is incomplete or duplicated.')
    expected_steps = list(range(100, expected + 1, 100))
    if expected not in expected_steps:
        expected_steps.append(expected)
    if [r['step'] for r in result['monitors']] != expected_steps:
        raise ValueError('Missing monitor reading.')
    best = min(result['monitors'], key=lambda r: (r['log_loss'], r['step']))
    if result['best'] != best:
        raise ValueError('Checkpoint selection differs from the preregistered monitor rule.')
    for step in (best['step'], expected):
        checkpoint = folder / f'checkpoint_{step:04d}'
        for name, digest in json.loads((checkpoint / 'receipt.json').read_text())['files'].items():
            if sha256(checkpoint / name) != digest:
                raise ValueError('Selected or final checkpoint changed.')
    return result, min(pairs, best['step'] * 4)


def run(bundle, output, minutes=360):
    gpu = require_a100()
    manifest, pairs = check_data(bundle)
    sources = {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')}
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + minutes * 60
    plan = {'gpu': gpu, 'model': manifest['model'], 'revision': manifest['revision'],
            'training_pairs': pairs, 'presentations': pairs * 2, 'updates': math.ceil(pairs / 4),
            'epochs': 1, 'monitor_pairs': 400, 'learning_rate': 2e-5, 'seed': 20261007,
            'worker_limits_minutes': {'probe': 20, 'training': 300, 'baseline': 20, 'selected': 20},
            'overall_minutes': minutes, 'automatic_retries': 0,
            'bundle_sha256': sha256(bundle / 'manifest.json'), 'source_sha256': sources,
            'selection': 'Lowest complete monitor log-loss, ties prefer the earlier update.',
            'calibration_and_test': 'not used; absent from the package'}
    save_json(output / 'full_run_plan.json', plan)

    def stage(name, mode, arguments, ceiling):
        remaining = min(ceiling, math.floor((deadline - time.monotonic()) / 60))
        if remaining < (10 if mode == 'train' else 1):
            raise TimeoutError('Overall compute budget exhausted; no automatic retry.')
        if {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')} != sources:
            raise ValueError('Worker source changed during the experiment.')
        verify_bundle(bundle)
        save_json(output / 'status.json', {'status': 'running', 'stage': name, 'maximum_remaining_minutes': remaining})
        print(json.dumps({'stage': name, 'minutes': remaining}), flush=True)
        if mode == 'train':
            arguments += ['--wall-minutes', str(remaining)]
        bounded(mode + '.py', arguments, output / name, remaining)

    try:
        shutil.copytree(Path(__file__).parent, output / 'source_snapshot', ignore=shutil.ignore_patterns('__pycache__', '*.zip', '*.ipynb'))
        shutil.copytree(bundle, output / 'bundle_snapshot')
        stage('probe', 'probe', ['--bundle', str(bundle)], 20)
        probe = json.loads((output / 'probe/result.json').read_text())
        verify_probe(probe, bundle, sources)
        stage('training', 'train', training_arguments(bundle, pairs), 300)
        trained, selected_exposure = verify_completed_training(output / 'training', pairs)
        selected = output / 'training' / f"checkpoint_{trained['best']['step']:04d}"
        save_json(output / 'selection.json', {'checkpoint': str(selected), 'best': trained['best'],
                  'selected_checkpoint_training_pairs_seen': selected_exposure, 'full_pass_pairs': pairs,
                  'final_checkpoint': str(output / 'training' / f"checkpoint_{plan['updates']:04d}")})
        stage('baseline_a100', 'evaluate', ['--bundle', str(bundle)], 20)
        stage('selected_a100', 'evaluate', ['--bundle', str(bundle), '--checkpoint', str(selected),
              '--baseline', str(output / 'baseline_a100')], 20)
        report = json.loads((output / 'selected_a100/full_report.json').read_text())
        summary = {'status': 'completed', 'training_pairs': pairs, 'updates': plan['updates'],
                   'selected_checkpoint': str(selected), 'selected_pairs_seen': selected_exposure,
                   'accuracy': report['views']['averaged'], 'development_screen': report['development_screen'],
                   'seconds': time.monotonic() - started,
                   'interpretation': 'Development validation versus unchanged A100 Qwen; not calibrated final-test performance.'}
        save_json(output / 'summary.json', summary)
        save_json(output / 'status.json', {'status': 'completed', 'stage': 'reports_saved'})
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException as error:
        save_json(output / 'status.json', {'status': 'failed', 'type': type(error).__name__, 'message': str(error),
                  'seconds': time.monotonic() - started, 'automatic_retries': 0})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--minutes', type=int, default=360)
    args = parser.parse_args()
    if not 30 <= args.minutes <= 360:
        parser.error('Use a compute ceiling between 30 and 360 minutes.')
    run(args.bundle, args.output, args.minutes)
