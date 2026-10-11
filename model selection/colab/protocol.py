"""Portable, label-safe helpers for the corrected judge experiments."""
import hashlib
import json
import math
import random
from pathlib import Path


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def vote_probability(votes):
    values = [v['strength'] for v in votes]
    if not 2 <= len(values) <= 6 or any(type(v) is not int or v not in (-3, -2, -1, 1, 2, 3) for v in values):
        raise ValueError('Require 2–6 nonzero signed votes; never guess missing vote targets.')
    return sum(v < 0 for v in values) / len(values)


def target_probability(row, method, swapped=False):
    if method == 'votes':
        probability = row['vote_p_a']
    elif method in ('hard', 'smooth') and row['target'] in ('A', 'B'):
        probability = float(row['target'] == 'A')
        if method == 'smooth':
            probability = .9 * probability + .1 * .5
    else:
        raise ValueError('Hard/smoothed targets require clear labels; ties require vote targets.')
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError('Invalid target probability.')
    return 1 - probability if swapped else probability


def length_weights(rows, balanced):
    counts = {value: sum(r['shorter_won'] is value for r in rows) for value in (True, False)}
    if not balanced:
        return {r['id']: 1.0 for r in rows}
    if min(counts.values()) == 0:
        raise ValueError('Balancing needs both shorter- and longer-winner examples.')
    total = sum(counts.values())
    return {r['id']: total / (2 * counts[r['shorter_won']]) if r['shorter_won'] is not None else 1.0 for r in rows}


def training_weights(rows, balanced=False, strength=False):
    weights = length_weights(rows, balanced)
    if strength:
        if any(r['target'] not in ('A', 'B') or r['strength'] not in (1, 2, 3) for r in rows):
            raise ValueError('Strength weighting requires clear labels with strength 1, 2 or 3.')
        weights = {r['id']: weights[r['id']] * r['strength'] for r in rows}
        mean = sum(weights.values()) / len(weights)
        weights = {key: value / mean for key, value in weights.items()}
    return weights


def epoch_batches(rows, epochs, seed, shuffle=True):
    if not rows or epochs < 1 or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Require nonempty unique pairs and at least one epoch.')
    batches = []
    for epoch in range(epochs):
        ordered = list(rows)
        # Preserve the recorded first pass; later permutations use an independent RNG.
        if shuffle and epoch:
            random.Random(f'{seed}:epoch:{epoch}').shuffle(ordered)
        batches.extend(ordered[i:i+4] for i in range(0, len(ordered), 4))
    return batches


def assert_separate(training, monitor, validation):
    groups = [{r['group_id'] for r in rows} for rows in (training, monitor, validation)]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError('Training, selection monitor, and validation groups must be disjoint.')
    for rows in (training, monitor, validation):
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate pair identity.')


def learning_rate(index, peak, warmup, updates):
    if not 0 <= index < updates or not 1 < warmup < updates - 1 or not 0 < peak <= 5e-5:
        raise ValueError('Invalid bounded learning-rate schedule.')
    if index < warmup:
        return peak / warmup + (peak - peak / warmup) * index / (warmup - 1)
    fraction = (index - warmup) / (updates - warmup - 1)
    return peak * (.1 + .9 * (1 + math.cos(math.pi * fraction)) / 2)


def basic_metrics(pairs):
    clear = [r for r in pairs if r['target'] in ('A', 'B')]
    if not clear:
        raise ValueError('Require clear labels for monitor log-loss.')
    correct = lambda r: (r['p_a'] > .5 and r['target'] == 'A') or (r['p_a'] < .5 and r['target'] == 'B')
    losses = [-math.log(max(1e-12, min(1-1e-12, r['p_a'] if r['target'] == 'A' else 1-r['p_a']))) for r in clear]
    return {'pairs': len(pairs), 'clear_pairs': len(clear), 'accuracy': sum(map(correct, clear))/len(clear), 'log_loss': sum(losses)/len(clear)}


def coverage_curve(pairs):
    clear = [r for r in pairs if r['target'] in ('A', 'B')]
    if not clear:
        raise ValueError('Coverage requires clear-labelled pairs.')
    confidence = sorted((abs(r['p_a']-.5) for r in clear), reverse=True)
    output = []
    for fraction in (1.0, .9, .75, .5):
        threshold = confidence[max(1, round(len(clear)*fraction))-1]
        kept = [r for r in clear if abs(r['p_a']-.5) >= threshold]
        output.append({'requested_coverage': fraction, 'actual_coverage': len(kept)/len(clear),
                       'margin_threshold': threshold, **basic_metrics(kept)})
    return output


def verify_bundle(folder):
    folder = Path(folder)
    manifest = json.loads((folder/'manifest.json').read_text())
    for name, expected in manifest['files_sha256'].items():
        path = (folder/name).resolve()
        if not path.is_relative_to(folder.resolve()) or sha256(path) != expected:
            raise ValueError('Bundle file changed or escapes its folder: '+name)
    return manifest
