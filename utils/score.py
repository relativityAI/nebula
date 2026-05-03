import numpy as np


def higher_score(x, midpoint):
    return 1 / (1 + np.exp(-8 * (x - midpoint)))


def lower_score(x, midpoint):
    return 1 / (1 + np.exp(8 * (x - midpoint)))


def sweet_score(x, center, width):
    return np.exp(-((x - center) ** 2) / (2 * width**2))


def score_metric(x, cfg):
    t = cfg["type"]
    if t == "higher":
        return float(higher_score(x, cfg["midpoint"]))
    elif t == "lower":
        return float(lower_score(x, cfg["midpoint"]))
    elif t == "sweet":
        return float(sweet_score(x, cfg["center"], cfg["width"]))
    return 0.0


def score_metrics(metrics, thresholds):
    total, wsum = 0.0, 0.0
    scores = {}
    for k, cfg in thresholds.items():
        if k not in metrics or metrics[k] is None or np.isnan(metrics[k]):
            continue
        s = score_metric(metrics[k], cfg)
        scores[k] = round(s, 3)
        total += s * cfg["weight"]
        wsum += cfg["weight"]
    return {
        "composite_score": round(total / wsum, 3) if wsum else None,
        "metrics_used": len(scores),
        "metric_scores": scores,
    }
