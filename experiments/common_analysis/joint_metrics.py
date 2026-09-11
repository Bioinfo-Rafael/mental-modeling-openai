"""Shared pure numerical functions, moved unchanged from Exp.03_1 analysis."""
from __future__ import annotations
import hashlib
import numpy as np


def stable_seed(seed: int, key: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "big")


def wrap_to_pi(value):
    return (np.asarray(value) + np.pi) % (2 * np.pi) - np.pi


def theta(state) -> float | None:
    cosine, sine = state[:2]
    if np.hypot(cosine, sine) == 0:
        return None  # atan2(0,0) is not a physically defined angle.
    return float(np.arctan2(sine, cosine))


def angular_pair(record: dict, component: str) -> tuple[float | None, float, str]:
    early, late = np.asarray(record["early"]), np.asarray(record["late"])
    a, b = theta(early), theta(late)
    if a is None or b is None:
        return None, float("nan"), "zero_norm_ground_truth"
    gt = float(wrap_to_pi(b - a))
    prediction = record["components"][component]
    if not prediction["ok"]:
        return None, gt, "component_parse_failure"
    target = np.asarray(prediction["value"], dtype=float)
    next_state = record["metric"] == "next-state"
    if component == "state_delta":
        target = early + target if next_state else late - target
    angle = theta(target)
    if angle is None:
        return None, gt, "zero_norm_reconstructed_state"
    predicted_change = angle - a if next_state else b - angle
    return float(wrap_to_pi(predicted_change)), gt, ""


def pearson(prediction, truth) -> tuple[float | None, str]:
    if len(prediction) < 2:
        return None, "fewer_than_two_pairs"
    p, g = np.asarray(prediction, float), np.asarray(truth, float)
    if np.ptp(p) == 0 or np.ptp(g) == 0:
        return None, "constant_vector"
    return float(np.corrcoef(p, g)[0, 1]), ""


def cosine(prediction, truth) -> tuple[float | None, str]:
    if not len(prediction):
        return None, "no_valid_pairs"
    p, g = np.asarray(prediction, float), np.asarray(truth, float)
    denominator = np.linalg.norm(p) * np.linalg.norm(g)
    if denominator == 0:
        return None, "zero_norm_vector"
    return float(np.clip(np.dot(p, g) / denominator, -1, 1)), ""


def nrmse_statistics(prediction, truth, widths, repetitions: int, seed: int, *, angular=False) -> dict:
    """All dimensions share paired query resamples, including the macro estimator."""
    p, g = np.asarray(prediction, float), np.asarray(truth, float)
    if p.ndim == 1:
        p, g = p[:, None], g[:, None]
    errors = p - g
    if angular:
        errors = wrap_to_pi(errors)
    scaled_square = (errors / np.asarray(widths)) ** 2
    point = np.sqrt(np.mean(scaled_square, axis=0))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(p), size=(repetitions, len(p)))
    boot = np.sqrt(np.mean(scaled_square[indices], axis=1))
    macro_boot = boot.mean(axis=1)
    return {"point": point, "bootstrap_mean": boot.mean(axis=0), "bootstrap_std": boot.std(axis=0, ddof=1),
            "macro": float(point.mean()), "macro_bootstrap_mean": float(macro_boot.mean()),
            "macro_bootstrap_std": float(macro_boot.std(ddof=1))}
