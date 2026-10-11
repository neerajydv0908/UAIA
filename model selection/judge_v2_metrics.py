"""Full-validation metrics, grouped uncertainty, and predeclared comparisons."""
import json
import math
import sys
from pathlib import Path

import numpy as np
from baseline import read_rows, save_json
from judge_v2 import JOBS, probabilities
from train_trial import collect_predictions


def choice(p):
    return "A" if p > 0.5 else ("B" if p < 0.5 else "tie_preference")


def combine(sample, scores):
    expected = {(row["id"], swapped) for row in sample for swapped in (False, True)}
    lookup = {}
    for row in scores:
        key = (row["id"], row["swapped"])
        if type(row["swapped"]) is not bool or key not in expected or key in lookup:
            raise ValueError("Unknown, duplicate, or malformed score identity.")
        if row["status"] == "completed":
            p, three = probabilities(row["logits"])
            if abs(p - row["p_display_a"]) > 1e-12 or not np.allclose(three, row["three_way_prefix_probs"], atol=1e-12, rtol=0):
                raise ValueError("Saved probabilities do not match their logits.")
        lookup[key] = row
    combined = []
    for row in sample:
        first, second = (lookup.get((row["id"], swapped), {"status": "not_run"}) for swapped in (False, True))
        complete = first["status"] == second["status"] == "completed"
        a, b = (len(row["input"]["answers"][key]) for key in ("A", "B"))
        entry = {"id": row["id"], "group_id": row["group_id"], "target": row["target"], "strength": row["strength"],
                 "shorter_won": row["target"] in ("A", "B") and a != b and ((a < b) == (row["target"] == "A")),
                 "status": "completed" if complete else "unresolved", "order_statuses": [first["status"], second["status"]]}
        if complete:
            original, swapped = first["p_display_a"], 1.0 - second["p_display_a"]
            entry.update(original=original, swapped=swapped, averaged=(original + swapped) / 2,
                         agrees=choice(original) == choice(swapped), position_gap=abs(original - swapped),
                         tie_prefix_original=first["three_way_prefix_probs"][2], tie_prefix_swapped=second["three_way_prefix_probs"][2])
        combined.append(entry)
    return combined


def weighted_metrics(y, p, weights):
    denom = weights.sum(axis=1)
    correct = ((p > 0.5) & (y == 1)) | ((p < 0.5) & (y == 0))
    clipped = np.clip(p, 1e-12, 1 - 1e-12)
    losses = -(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))
    accuracy = weights @ correct.astype(float) / denom
    logloss = weights @ losses / denom
    order = np.argsort(p, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(p[order])) + 1]
    positives = np.add.reduceat(weights[:, order] * y[order], starts, axis=1)
    negatives = np.add.reduceat(weights[:, order] * (1 - y[order]), starts, axis=1)
    wins = (positives * (np.cumsum(negatives, axis=1) - negatives / 2)).sum(axis=1)
    denominator = positives.sum(axis=1) * negatives.sum(axis=1)
    auc = np.divide(wins, denominator, out=np.full(len(weights), np.nan), where=denominator > 0)
    return np.stack((accuracy, logloss, auc), axis=1)


def bootstrap_metrics(rows, field, replicates, seed, other=None):
    y = np.array([r["target"] == "A" for r in rows], dtype=float)
    p = np.array([r[field] for r in rows])
    groups, inverse = np.unique([r["group_id"] for r in rows], return_inverse=True)
    one = np.ones((1, len(rows)))
    estimate = weighted_metrics(y, p, one)[0]
    q = np.array([r[other] for r in rows]) if other else None
    if q is not None:
        estimate -= weighted_metrics(y, q, one)[0]
    rng, draws = np.random.default_rng(seed), []
    for start in range(0, replicates, 250):
        counts = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=min(250, replicates - start))
        weights = counts[:, inverse]
        values = weighted_metrics(y, p, weights)
        if q is not None:
            values -= weighted_metrics(y, q, weights)
        draws.append(values)
    distribution = np.concatenate(draws)
    result = {}
    for index, name in enumerate(("accuracy", "log_loss", "auc")):
        valid = distribution[:, index][np.isfinite(distribution[:, index])]
        value = float(estimate[index]) if np.isfinite(estimate[index]) else None
        result[name] = {"value": value, "ci95": np.quantile(valid, [0.025, 0.975]).tolist() if len(valid) else None,
                        "valid_bootstrap_replicates": len(valid)}
    return result


def measure(rows, field, settings, rule=False):
    complete = [r for r in rows if r["status"] == "completed"]
    result = {"planned": len(rows), "completed": len(complete), "groups": len({r["group_id"] for r in rows})}
    if not rows or len(rows) != len(complete):
        return {**result, "status": "empty" if not rows else "incomplete", "metrics": None}
    result.update(status="completed", matches=sum(choice(r[field]) == r["target"] for r in rows),
                  metrics=bootstrap_metrics(rows, field, settings["bootstrap_replicates"], settings["bootstrap_seed"]))
    if rule:
        result["metrics"]["log_loss"] = {"value": None, "ci95": None, "reason": "Hard rule supplies no estimated probability; do not invent one."}
    return result


def diagnostic(rows, settings):
    good = [r for r in rows if r["status"] == "completed"]
    if len(good) != len(rows):
        return {"status": "incomplete", "completed_pairs": len(good), "planned_pairs": len(rows)}
    ties = [r for r in rows if r["target"] == "tie_preference"]
    clear = [r for r in rows if r["target"] != "tie_preference"]
    bands = []
    for band in settings["tie_bands"]:
        recalled = sum(abs(r["averaged"] - 0.5) <= band for r in ties)
        false_ties = sum(abs(r["averaged"] - 0.5) <= band for r in clear)
        recall, coverage = recalled / len(ties), 1 - false_ties / len(clear)
        bands.append({"band": band, "tie_matches": recalled, "tie_recall": recall, "clear_abstentions": false_ties,
                      "clear_coverage": coverage, "balanced_tie_detection": (recall + coverage) / 2})
    eligible = [r for r in bands if r["clear_coverage"] >= settings["minimum_clear_coverage_for_tie_band"]]
    selected = max(eligible, key=lambda r: (r["balanced_tie_detection"], -r["band"])) if eligible else None
    return {"status": "completed", "order_agreement": sum(r["agrees"] for r in rows), "pairs": len(rows),
            "mean_order_probability_gap": float(np.mean([r["position_gap"] for r in rows])),
            "tie_pairs": len(ties), "tie_mean_p_a": float(np.mean([r["averaged"] for r in ties])),
            "tie_mean_absolute_distance_from_half": float(np.mean([abs(r["averaged"] - 0.5) for r in ties])),
            "tie_mean_prefix_probability": float(np.mean([r[key] for r in ties for key in ("tie_prefix_original", "tie_prefix_swapped")])),
            "bands": bands, "selected_validation_band": selected, "band_status": "validation_fitted_diagnostic_not_calibrated_or_final"}


def summary(rows, settings, rule=False):
    clear = [r for r in rows if r["target"] != "tie_preference"]
    views = {field: measure(clear, field, settings, rule) for field in ("original", "swapped", "averaged")}
    slices = {"strength_" + str(level): measure([r for r in clear if r["strength"] == level], "averaged", settings, rule) for level in (1, 2, 3)}
    slices["shorter_winner"] = measure([r for r in clear if r["shorter_won"]], "averaged", settings, rule)
    return {"views": views, "slices": slices, "diagnostics": diagnostic(rows, settings) if not rule else {"status": "not_applicable_hard_rule"},
            "rule": rule, "auc_interpretation": "Hard-decision ranking" if rule else "A-versus-B conditional probability ranking"}


def mcnemar(before, after):
    gain = sum(not a and b for a, b in zip(before, after))
    loss = sum(a and not b for a, b in zip(before, after))
    n = gain + loss
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(gain, loss) + 1)) / 2**n) if n else 1.0
    return {"improved": gain, "regressed": loss, "discordant": n, "exact_two_sided_p": p}


def compare(before, after, settings, before_field="averaged", after_field="averaged"):
    if [(r["id"], r["target"], r["group_id"]) for r in before] != [(r["id"], r["target"], r["group_id"]) for r in after]:
        raise ValueError("Paired comparison has different examples or labels.")
    if any(r["status"] != "completed" for r in before + after):
        return {"status": "incomplete"}
    rows = [{**a, "before": a[before_field], "after": b[after_field]} for a, b in zip(before, after) if a["target"] in ("A", "B")]
    delta = bootstrap_metrics(rows, "after", settings["bootstrap_replicates"], settings["bootstrap_seed"], other="before")
    canonical = {}
    for row in sorted(rows, key=lambda r: r["id"]):
        canonical.setdefault(row["group_id"], row)
    selected = list(canonical.values())
    paired_test = mcnemar([choice(r["before"]) == r["target"] for r in selected], [choice(r["after"]) == r["target"] for r in selected])
    full_counts = mcnemar([choice(r["before"]) == r["target"] for r in rows], [choice(r["after"]) == r["target"] for r in rows])
    full_counts.pop("exact_two_sided_p")
    return {"status": "completed", "delta_after_minus_before": delta, "all_clear_pair_change_counts": full_counts,
            "mcnemar": {**paired_test, "independent_groups": len(selected), "rule": "Lexically first ID per question group; main accuracy and grouped bootstrap retain all pairs."}}


def holm(comparisons):
    eligible = [(key, r["mcnemar"]["exact_two_sided_p"]) for key, r in comparisons.items() if r["status"] == "completed"]
    previous = 0.0
    for index, (key, p) in enumerate(sorted(eligible, key=lambda item: item[1])):
        previous = max(previous, min(1.0, p * (len(eligible) - index)))
        comparisons[key]["mcnemar"]["holm_adjusted_p"] = previous


def format_passes(comparison, settings):
    if comparison["status"] != "completed":
        return {"accepted": False, "reason": "incomplete"}
    delta = comparison["delta_after_minus_before"]
    acc, loss = delta["accuracy"], delta["log_loss"]
    accuracy_path = acc["value"] >= settings["minimum_accuracy_gain"] and acc["ci95"][0] > 0 and comparison["mcnemar"]["holm_adjusted_p"] < settings["familywise_alpha"] and loss["value"] <= 0
    probability_path = loss["value"] <= -settings["minimum_logloss_reduction"] and loss["ci95"][1] < 0 and acc["value"] >= 0
    return {"accepted": accuracy_path or probability_path, "accuracy_path": accuracy_path, "probability_path": probability_path,
            "interpretation": "Development format selection only; probability path is not a significant accuracy-gain claim."}


def analyze(output):
    plan = json.loads((output / "plan.json").read_text())
    sample = read_rows(output / "sample.jsonl")
    settings, paired, summaries, damaged = plan["statistics"], {}, {}, {}
    for job in JOBS:
        path = output / job / "scores.jsonl"
        scores, damaged[job] = collect_predictions(path)
        paired[job] = combine(sample, scores)
        summaries[job] = summary(paired[job], settings)
        save_json(output / job / "paired_scores.json", paired[job])
    for rule in ("always_B", "longer_answer"):
        records = []
        for row in sample:
            a, b = (len(row["input"]["answers"][key]) for key in ("A", "B"))
            p = 0.0 if rule == "always_B" else (1.0 if a > b else (0.0 if a < b else 0.5))
            records.append({"id": row["id"], "group_id": row["group_id"], "target": row["target"], "strength": row["strength"],
                            "status": "completed", "original": p, "swapped": p, "averaged": p,
                            "shorter_won": row["target"] in ("A", "B") and a != b and ((a < b) == (row["target"] == "A"))})
        summaries[rule] = summary(records, settings, rule=True)
    comparisons = {after + "_vs_" + before: compare(paired[before], paired[after], settings) for before, after in settings["primary_comparisons"]}
    holm(comparisons)
    averaging = {job: compare(rows, rows, settings, before_field="original", after_field="averaged") for job, rows in paired.items()}
    candidate = format_passes(comparisons["baseline_v2_plain_vs_baseline_v2_json"], settings)
    execution = json.loads((output / "execution.json").read_text())
    complete = execution["status"] == "completed" and not any(damaged.values()) and all(r["status"] == "completed" for rows in paired.values() for r in rows)
    if not complete:
        candidate.update(accepted=False, reason="Incomplete milestone; no format accepted.")
    save_json(output / "metrics.json", {"milestone_complete": complete, "damaged_tails": damaged, "versions": summaries, "primary_comparisons": comparisons, "averaging_comparisons_exploratory": averaging,
              "format_selection": candidate, "selected_format": "plain" if candidate["accepted"] else "json",
              "calibration_status": "not_run_wait_for_final_model", "final_test_status": "unopened",
              "interval_interpretation": "Pointwise 95% grouped percentile intervals; dependent orientations stay in one pair. Repeated development, not final-test evidence."})
    print("V2 metrics saved:", output / "metrics.json", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Use judge_v2_metrics.py RESULT_FOLDER.")
    analyze(Path(sys.argv[1]).resolve())
