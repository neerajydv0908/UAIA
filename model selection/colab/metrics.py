"""Portable copies of the established grouped-bootstrap metrics; inference is separate."""
import math
import numpy as np

def choice(p):
    return "A" if p > 0.5 else ("B" if p < 0.5 else "tie_preference")


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


def format_passes(comparison, settings):
    if comparison["status"] != "completed":
        return {"accepted": False, "reason": "incomplete"}
    delta = comparison["delta_after_minus_before"]
    acc, loss = delta["accuracy"], delta["log_loss"]
    accuracy_path = acc["value"] >= settings["minimum_accuracy_gain"] and acc["ci95"][0] > 0 and comparison["mcnemar"]["holm_adjusted_p"] < settings["familywise_alpha"] and loss["value"] <= 0
    probability_path = loss["value"] <= -settings["minimum_logloss_reduction"] and loss["ci95"][1] < 0 and acc["value"] >= 0
    return {"accepted": accuracy_path or probability_path, "accuracy_path": accuracy_path, "probability_path": probability_path,
            "interpretation": "Development format selection only; probability path is not a significant accuracy-gain claim."}

