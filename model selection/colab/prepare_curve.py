"""Version a larger training-side monitor without opening reserved dataset splits."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
from protocol import assert_separate, read_rows, save_json, sha256, verify_bundle


def stratum(row):
    return (row['strength'], row['shorter_won'], row['multi_turn'])


def enlarge_monitor(training, monitor, pool_ids, size=400, seed=20261007):
    if size <= len(monitor) or any(r['target'] not in ('A', 'B') for r in monitor):
        raise ValueError('Enlarge the existing clear-label monitor; never recycle it into training.')
    assert_separate(training, monitor, [])
    pool_groups = {r['group_id'] for r in training if r['id'] in set(pool_ids)}
    clear = [r for r in training + monitor if r['target'] in ('A', 'B')]
    mix = Counter(map(stratum, clear))
    wanted = {key: size * count / len(clear) for key, count in mix.items()}
    have = Counter(map(stratum, monitor))
    groups = defaultdict(list)
    for row in training:
        groups[row['group_id']].append(row)
    candidates = defaultdict(list)
    for group, rows in groups.items():
        eligible = [r for r in rows if r['target'] in ('A', 'B')]
        if group not in pool_groups and len(eligible) == 1:
            candidates[stratum(eligible[0])].append(eligible[0])
    for rows in candidates.values():
        rows.sort(key=lambda r: hashlib.sha256(f"{seed}:{r['group_id']}:{r['id']}".encode()).hexdigest(), reverse=True)
    selected = list(monitor)
    while len(selected) < size:
        available = [key for key, rows in candidates.items() if rows]
        if not available:
            raise ValueError('Not enough unused groups outside the preserved 2,000-pair pool.')
        key = min(available, key=lambda k: (have[k] - wanted[k], str(k)))
        selected.append(candidates[key].pop())
        have[key] += 1
    excluded = {r['group_id'] for r in selected}
    remaining = [r for r in training if r['group_id'] not in excluded]
    assert_separate(remaining, selected, [])
    if not set(pool_ids) <= {r['id'] for r in remaining}:
        raise ValueError('The original pool must remain intact.')
    return remaining, selected


def mix_summary(rows):
    clear = [r for r in rows if r['target'] in ('A', 'B')]
    return {'pairs': len(rows), 'groups': len({r['group_id'] for r in rows}),
            'clear': len(clear), 'ties': len(rows) - len(clear),
            'shorter_winners': sum(r['shorter_won'] is True for r in clear),
            'strength': dict(Counter(str(r['strength']) for r in clear)),
            'multi_turn': sum(r['multi_turn'] for r in clear)}


def prepare(source, output, size=400, seed=20261007):
    original = verify_bundle(source)
    old_training = read_rows(source / 'training.jsonl')
    old_monitor = read_rows(source / 'monitor.jsonl')
    training, monitor = enlarge_monitor(old_training, old_monitor, original['pool2000_ids'], size, seed)
    assert_separate(training, monitor, [{'id': g, 'group_id': g} for g in original['validation_groups']])
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in (('training', training), ('monitor', monitor)):
        (output / f'{name}.jsonl').write_text(''.join(json.dumps(r, separators=(',', ':')) + '\n' for r in rows))
    # Preserve the existing public development validation bytes; never access calibration/test.
    shutil.copy2(source / 'validation.jsonl', output / 'validation.jsonl')
    audit = {'source_bundle_sha256': sha256(source / 'manifest.json'), 'seed': seed,
             'monitor_selection': 'Retain original holdout; fill deficits in training-side strength/length/turn strata.',
             'training_before': mix_summary(old_training), 'training': mix_summary(training),
             'monitor_before': mix_summary(old_monitor), 'monitor': mix_summary(monitor),
             'old_monitor_retained': True, 'original_pool_preserved': True,
             'validation_bytes_unchanged': sha256(source / 'validation.jsonl') == sha256(output / 'validation.jsonl'),
             'calibration_and_test_opened': False}
    save_json(output / 'data_audit.json', audit)
    counts = original['counts'] | {'eligible_after_holdout': len(training),
             'eligible_clear_after_holdout': audit['training']['clear'],
             'eligible_ties_after_holdout': audit['training']['ties'],
             'monitor_pairs': len(monitor), 'monitor_groups': audit['monitor']['groups']}
    manifest = original | {'counts': counts, 'monitor_groups': sorted({r['group_id'] for r in monitor}),
             'parent_bundle_sha256': sha256(source / 'manifest.json'),
             'monitor_version': {'pairs': size, 'seed': seed, 'method': audit['monitor_selection']},
             'files_sha256': {p.name: sha256(p) for p in output.iterdir() if p.is_file()}}
    save_json(output / 'manifest.json', manifest)
    verify_bundle(output)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path); parser.add_argument('output', type=Path)
    parser.add_argument('--monitor-pairs', type=int, default=400)
    parser.add_argument('--seed', type=int, default=20261007)
    args = parser.parse_args()
    prepare(args.source, args.output, args.monitor_pairs, args.seed)
