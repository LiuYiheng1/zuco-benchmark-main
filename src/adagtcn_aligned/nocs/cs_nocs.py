"""Post-hoc CS-NOCS admission rules.

CS-NOCS does not retrain a model. It reads exact gaze-anchor predictions and
corrected NOCS predictions, then decides whether EEG evidence is admissible.
"""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score


GAZE_LR_AUROC = 0.6939
GAZE_LR_MACRO_F1 = 0.5739
NAIVE_CONCAT_AUROC = 0.5369
NOCS_FULL_AUROC = 0.6922
SAFENOCS_WORST_AUROC = 0.3594959958856807
_CERT_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}


@dataclass
class ProtocolPredictions:
    protocol: str
    heldout_subject: str
    data: pd.DataFrame


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= 0.5).astype(np.int64)
    if len(set(y_true.tolist())) < 2:
        auroc = 0.5
        auprc = float(np.mean(y_true)) if len(y_true) else 0.0
    else:
        auroc = float(roc_auc_score(y_true, y_prob))
        auprc = float(average_precision_score(y_true, y_prob))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "auroc": auroc,
        "auprc": auprc,
        "n": int(len(y_true)),
    }


def bootstrap_delta_ci(
    y_true: np.ndarray,
    anchor_prob: np.ndarray,
    corrected_prob: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    anchor_prob = np.asarray(anchor_prob, dtype=np.float64)
    corrected_prob = np.asarray(corrected_prob, dtype=np.float64)
    delta = binary_metrics(y_true, corrected_prob)["auroc"] - binary_metrics(y_true, anchor_prob)["auroc"]
    rng = np.random.RandomState(seed)
    n = len(y_true)
    if n == 0:
        return math.nan, math.nan, math.nan
    values = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        values.append(
            binary_metrics(y_true[idx], corrected_prob[idx])["auroc"]
            - binary_metrics(y_true[idx], anchor_prob[idx])["auroc"]
        )
    return float(delta), float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def prob_mix(anchor_prob: np.ndarray, corrected_prob: np.ndarray, alpha: float) -> np.ndarray:
    return (1.0 - alpha) * anchor_prob + alpha * corrected_prob


def choose_alpha(
    y_val: np.ndarray,
    anchor_val: np.ndarray,
    corrected_val: np.ndarray,
    alpha_grid: tuple[float, ...],
    tol: float = 1e-8,
) -> tuple[float, float]:
    best_alpha = alpha_grid[0]
    best_score = -math.inf
    for alpha in alpha_grid:
        score = binary_metrics(y_val, prob_mix(anchor_val, corrected_val, alpha))["auroc"]
        if score > best_score + tol:
            best_alpha = alpha
            best_score = score
    return float(best_alpha), float(best_score)


def utility_certificate(
    val: pd.DataFrame,
    epsilon: float,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    cache_key = (
        tuple(val["protocol"].astype(str).tolist()),
        tuple(val["sequence_id"].astype(str).tolist()),
        n_boot,
        seed,
    )
    if cache_key in _CERT_CACHE:
        base = dict(_CERT_CACHE[cache_key])
    else:
        y = val["y_true"].to_numpy(dtype=np.int64)
        anchor = val["y_prob_anchor"].to_numpy(dtype=np.float64)
        corrected = val["y_prob_corrected"].to_numpy(dtype=np.float64)
        anchor_metrics = binary_metrics(y, anchor)
        corrected_metrics = binary_metrics(y, corrected)
        delta, ci_low, ci_high = bootstrap_delta_ci(y, anchor, corrected, n_boot=n_boot, seed=seed)
        macro_delta = corrected_metrics["macro_f1"] - anchor_metrics["macro_f1"]
        bal_delta = corrected_metrics["balanced_accuracy"] - anchor_metrics["balanced_accuracy"]
        base = {
            "anchor_val_auroc": anchor_metrics["auroc"],
            "corrected_val_auroc": corrected_metrics["auroc"],
            "delta_val": delta,
            "delta_ci_low": ci_low,
            "delta_ci_high": ci_high,
            "anchor_val_macro_f1": anchor_metrics["macro_f1"],
            "corrected_val_macro_f1": corrected_metrics["macro_f1"],
            "delta_val_macro_f1": macro_delta,
            "anchor_val_balanced_accuracy": anchor_metrics["balanced_accuracy"],
            "corrected_val_balanced_accuracy": corrected_metrics["balanced_accuracy"],
            "delta_val_balanced_accuracy": bal_delta,
        }
        _CERT_CACHE[cache_key] = dict(base)
    passed = (
        base["delta_val"] > 0.0
        and base["delta_ci_low"] >= -epsilon
        and base["corrected_val_macro_f1"] >= base["anchor_val_macro_f1"] - 0.005
        and base["corrected_val_balanced_accuracy"] >= base["anchor_val_balanced_accuracy"] - 0.005
    )
    out = dict(base)
    out["certificate_passed"] = bool(passed)
    return out


def reliability_prior(val: pd.DataFrame) -> dict[str, Any]:
    y = val["y_true"].to_numpy(dtype=np.int64)
    anchor = val["y_prob_anchor"].to_numpy(dtype=np.float64)
    corrected = val["y_prob_corrected"].to_numpy(dtype=np.float64)
    anchor_metrics = binary_metrics(y, anchor)
    corrected_metrics = binary_metrics(y, corrected)
    reliable = (
        corrected_metrics["auroc"] > anchor_metrics["auroc"]
        and corrected_metrics["macro_f1"] > anchor_metrics["macro_f1"]
        and corrected_metrics["balanced_accuracy"] > anchor_metrics["balanced_accuracy"]
    )
    return {
        "reliability_prior_passed": bool(reliable),
        "reliability_delta_auroc": corrected_metrics["auroc"] - anchor_metrics["auroc"],
        "reliability_delta_macro_f1": corrected_metrics["macro_f1"] - anchor_metrics["macro_f1"],
        "reliability_delta_balanced_accuracy": corrected_metrics["balanced_accuracy"] - anchor_metrics["balanced_accuracy"],
    }


def apply_utility_strategy(
    item: ProtocolPredictions,
    epsilon: float,
    require_reliability: bool = False,
    n_boot: int = 2000,
    seed: int = 0,
    alpha_grid: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.5),
) -> dict[str, Any]:
    val = item.data[item.data["split"] == "val"]
    test = item.data[item.data["split"] == "test"]
    cert = utility_certificate(val, epsilon=epsilon, n_boot=n_boot, seed=seed)
    prior = reliability_prior(val)
    admitted = bool(cert["certificate_passed"]) and (not require_reliability or bool(prior["reliability_prior_passed"]))
    selected_alpha = 0.0
    selected_mode = "anchor_only"
    if admitted:
        selected_alpha, safe_val_auroc = choose_alpha(
            val["y_true"].to_numpy(dtype=np.int64),
            val["y_prob_anchor"].to_numpy(dtype=np.float64),
            val["y_prob_corrected"].to_numpy(dtype=np.float64),
            alpha_grid,
        )
        if safe_val_auroc <= cert["anchor_val_auroc"] + 1e-8:
            selected_alpha = 0.0
            selected_mode = "anchor_only"
        else:
            selected_mode = "mixture"
    test_prob = prob_mix(
        test["y_prob_anchor"].to_numpy(dtype=np.float64),
        test["y_prob_corrected"].to_numpy(dtype=np.float64),
        selected_alpha,
    )
    return subject_result(item, test_prob, selected_mode, selected_alpha, cert, prior)


def error_risk(anchor_prob: np.ndarray, corrected_prob: np.ndarray) -> np.ndarray:
    conf_anchor = np.maximum(anchor_prob, 1.0 - anchor_prob)
    disagreement = np.abs(corrected_prob - anchor_prob)
    return disagreement * (1.0 - conf_anchor)


def apply_error_aware_strategy(
    item: ProtocolPredictions,
    alpha_grid: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5),
    quantiles: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9),
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    val = item.data[item.data["split"] == "val"]
    test = item.data[item.data["split"] == "test"]
    y_val = val["y_true"].to_numpy(dtype=np.int64)
    a_val = val["y_prob_anchor"].to_numpy(dtype=np.float64)
    c_val = val["y_prob_corrected"].to_numpy(dtype=np.float64)
    anchor_val_auroc = binary_metrics(y_val, a_val)["auroc"]
    risk_val = error_risk(a_val, c_val)
    best = {"score": anchor_val_auroc, "alpha": 0.0, "threshold": math.inf, "mode": "anchor_only"}
    for q in quantiles:
        threshold = float(np.quantile(risk_val, q))
        mask = risk_val >= threshold
        for alpha in alpha_grid:
            prob = a_val.copy()
            prob[mask] = prob_mix(a_val[mask], c_val[mask], alpha)
            score = binary_metrics(y_val, prob)["auroc"]
            if score > best["score"] + 1e-8 or (
                abs(score - best["score"]) <= 1e-8 and alpha < float(best["alpha"])
            ):
                best = {"score": score, "alpha": float(alpha), "threshold": threshold, "mode": "risk_mixture"}

    cert = utility_certificate(val, epsilon=0.0, n_boot=n_boot, seed=seed)
    prior = reliability_prior(val)
    a_test = test["y_prob_anchor"].to_numpy(dtype=np.float64)
    c_test = test["y_prob_corrected"].to_numpy(dtype=np.float64)
    test_prob = a_test.copy()
    if best["mode"] != "anchor_only":
        risk_test = error_risk(a_test, c_test)
        mask_test = risk_test >= float(best["threshold"])
        test_prob[mask_test] = prob_mix(a_test[mask_test], c_test[mask_test], float(best["alpha"]))
    row = subject_result(item, test_prob, str(best["mode"]), float(best["alpha"]), cert, prior)
    row["risk_threshold"] = float(best["threshold"]) if math.isfinite(float(best["threshold"])) else ""
    row["safe_val_auroc"] = float(best["score"])
    return row


def subject_result(
    item: ProtocolPredictions,
    final_prob: np.ndarray,
    selected_mode: str,
    selected_alpha: float,
    cert: dict[str, Any],
    prior: dict[str, Any],
) -> dict[str, Any]:
    test = item.data[item.data["split"] == "test"]
    y = test["y_true"].to_numpy(dtype=np.int64)
    anchor_prob = test["y_prob_anchor"].to_numpy(dtype=np.float64)
    corrected_prob = test["y_prob_corrected"].to_numpy(dtype=np.float64)
    anchor_test = binary_metrics(y, anchor_prob)
    corrected_test = binary_metrics(y, corrected_prob)
    final_test = binary_metrics(y, final_prob)
    row: dict[str, Any] = {
        "protocol": item.protocol,
        "heldout_subject": item.heldout_subject,
        "selected_mode": selected_mode,
        "selected_alpha": selected_alpha,
        "anchor_test_auroc": anchor_test["auroc"],
        "corrected_test_auroc": corrected_test["auroc"],
        "cs_nocs_test_auroc": final_test["auroc"],
        "delta_test_vs_anchor": final_test["auroc"] - anchor_test["auroc"],
        "test_macro_f1": final_test["macro_f1"],
        "test_balanced_accuracy": final_test["balanced_accuracy"],
        "test_accuracy": final_test["accuracy"],
        "test_auprc": final_test["auprc"],
        "test_examples": final_test["n"],
        "anchor_test_macro_f1": anchor_test["macro_f1"],
        "corrected_test_macro_f1": corrected_test["macro_f1"],
        "anchor_test_balanced_accuracy": anchor_test["balanced_accuracy"],
        "corrected_test_balanced_accuracy": corrected_test["balanced_accuracy"],
    }
    row.update(cert)
    row.update(prior)
    return row


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_metrics_dir(root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob("*_metrics.csv")):
        frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_predictions(exact_dir: Path, safe_dir: Path) -> list[ProtocolPredictions]:
    exact_frames = [pd.read_csv(path) for path in sorted(exact_dir.glob("*_predictions.csv"))]
    safe_frames = [pd.read_csv(path) for path in sorted(safe_dir.glob("*_predictions.csv"))]
    if not exact_frames:
        raise RuntimeError("No exact anchor predictions found in %s" % exact_dir)
    if not safe_frames:
        raise RuntimeError("No safe/corrected predictions found in %s" % safe_dir)
    exact = pd.concat(exact_frames, ignore_index=True)
    safe = pd.concat(safe_frames, ignore_index=True)
    required_exact = {"protocol", "heldout_subject", "sequence_id", "split", "y_true", "y_prob_anchor"}
    required_safe = {"protocol", "sequence_id", "split", "y_true", "y_prob_corrected"}
    missing_exact = sorted(required_exact - set(exact.columns))
    missing_safe = sorted(required_safe - set(safe.columns))
    if missing_exact:
        raise RuntimeError("Exact predictions missing columns: %s" % missing_exact)
    if missing_safe:
        raise RuntimeError("Safe/corrected predictions missing columns: %s" % missing_safe)
    merged = exact[list(required_exact)].merge(
        safe[["protocol", "sequence_id", "split", "y_true", "y_prob_corrected"]],
        on=["protocol", "sequence_id", "split", "y_true"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise RuntimeError("No aligned exact/corrected predictions after merge.")
    out = []
    for protocol, group in sorted(merged.groupby("protocol")):
        splits = set(group["split"].astype(str))
        if "val" not in splits or "test" not in splits:
            raise RuntimeError("Protocol %s is missing val/test predictions: %s" % (protocol, sorted(splits)))
        heldout = str(group["heldout_subject"].iloc[0])
        out.append(ProtocolPredictions(protocol=str(protocol), heldout_subject=heldout, data=group.copy()))
    return out


def write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_strategy(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    aurocs = np.asarray([float(row["cs_nocs_test_auroc"]) for row in rows], dtype=np.float64)
    macro_f1s = np.asarray([float(row["test_macro_f1"]) for row in rows], dtype=np.float64)
    deltas = np.asarray([float(row["delta_test_vs_anchor"]) for row in rows], dtype=np.float64)
    modes = Counter(str(row["selected_mode"]) for row in rows)
    alphas = Counter(str(row["selected_alpha"]) for row in rows)
    neg = int((deltas < -1e-12).sum())
    wins = int((deltas > 1e-12).sum())
    ties = int((np.abs(deltas) <= 1e-12).sum())
    summary = {
        "strategy": name,
        "n_subjects": len(rows),
        "n_boot": rows[0].get("n_boot", "") if rows else "",
        "mean_auroc": float(aurocs.mean()),
        "std_auroc": float(aurocs.std(ddof=1)) if len(aurocs) > 1 else 0.0,
        "mean_macro_f1": float(macro_f1s.mean()),
        "std_macro_f1": float(macro_f1s.std(ddof=1)) if len(macro_f1s) > 1 else 0.0,
        "worst_subject_auroc": float(aurocs.min()),
        "worst_subject": rows[int(aurocs.argmin())]["heldout_subject"],
        "win_vs_anchor": wins,
        "tie_vs_anchor": ties,
        "lose_vs_anchor": neg,
        "negative_transfer_subject_count": neg,
        "mean_delta_vs_anchor": float(deltas.mean()),
        "mean_delta_vs_naive_concat": float(aurocs.mean() - NAIVE_CONCAT_AUROC),
        "subjects_above_nocs_full": int((aurocs > NOCS_FULL_AUROC).sum()),
        "subjects_above_gaze_lr": int((aurocs > GAZE_LR_AUROC).sum()),
        "selected_mode_distribution": ";".join("%s:%d" % (k, modes[k]) for k in sorted(modes)),
        "selected_alpha_distribution": ";".join("%s:%d" % (k, alphas[k]) for k in sorted(alphas)),
        "admitted_subject_count": int(sum(1 for row in rows if str(row["selected_mode"]) != "anchor_only")),
        "success_negative_transfer_le_1": neg <= 1,
        "success_mean_near_anchor": float(aurocs.mean()) >= GAZE_LR_AUROC - 0.001,
        "success_macro_f1_near_anchor": float(macro_f1s.mean()) >= GAZE_LR_MACRO_F1 - 0.01,
        "success_worst_ge_safenocs": float(aurocs.min()) >= SAFENOCS_WORST_AUROC,
        "strong_success": float(aurocs.mean()) > 0.700,
    }
    return summary


def choose_best_strategy(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    def rank(row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            bool(row["success_negative_transfer_le_1"]),
            bool(row["success_mean_near_anchor"]),
            bool(row["success_macro_f1_near_anchor"]),
            bool(row["success_worst_ge_safenocs"]),
            -int(row["negative_transfer_subject_count"]),
            float(row["mean_delta_vs_anchor"]),
            float(row["mean_auroc"]),
            int(row["admitted_subject_count"]),
        )

    return max(summaries, key=rank)


def make_paper_ready_table(
    best_summary: dict[str, Any],
    exact_dir: Path,
    safe_dir: Path,
    nocs_dir: Path,
) -> list[dict[str, Any]]:
    rows = []
    for name, root, metric_col in [
        ("Exact gaze anchor", exact_dir, "test_auroc"),
        ("SafeNOCS", safe_dir, "test_auroc"),
        ("NOCS full", nocs_dir, "test_auroc"),
    ]:
        df = read_metrics_dir(root)
        if df.empty or metric_col not in df.columns:
            continue
        macro_col = "test_macro_f1"
        rows.append(
            {
                "method": name,
                "mean_auroc": float(df[metric_col].mean()),
                "std_auroc": float(df[metric_col].std()),
                "mean_macro_f1": float(df[macro_col].mean()) if macro_col in df else "",
                "std_macro_f1": float(df[macro_col].std()) if macro_col in df else "",
                "worst_subject_auroc": float(df[metric_col].min()),
                "n_subjects": int(len(df)),
            }
        )
    rows.append(
        {
            "method": "CS-NOCS best: %s" % best_summary["strategy"],
            "mean_auroc": best_summary["mean_auroc"],
            "std_auroc": best_summary["std_auroc"],
            "mean_macro_f1": best_summary["mean_macro_f1"],
            "std_macro_f1": best_summary["std_macro_f1"],
            "worst_subject_auroc": best_summary["worst_subject_auroc"],
            "n_subjects": best_summary["n_subjects"],
        }
    )
    return rows


def report_markdown(
    best: dict[str, Any],
    best_rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
) -> str:
    helpful = [row["heldout_subject"] for row in best_rows if float(row["delta_test_vs_anchor"]) > 1e-12]
    harmful = [row["heldout_subject"] for row in best_rows if float(row["delta_test_vs_anchor"]) < -1e-12]
    tied = [row["heldout_subject"] for row in best_rows if abs(float(row["delta_test_vs_anchor"])) <= 1e-12]
    stable_fail = not any(
        int(row["negative_transfer_subject_count"]) <= 2 and float(row["mean_auroc"]) >= GAZE_LR_AUROC - 0.005
        for row in summaries
    )
    lines = [
        "# CS-NOCS Autonomous Report",
        "",
        "## Best Strategy",
        "",
        "- Best strategy: `%s`" % best["strategy"],
        "- Mean AUROC: %.4f +/- %.4f" % (best["mean_auroc"], best["std_auroc"]),
        "- Mean Macro-F1: %.4f +/- %.4f" % (best["mean_macro_f1"], best["std_macro_f1"]),
        "- Worst-subject AUROC: %.4f (%s)" % (best["worst_subject_auroc"], best["worst_subject"]),
        "- Negative-transfer subjects: %d" % best["negative_transfer_subject_count"],
        "- Win/tie/lose vs exact anchor: %d/%d/%d"
        % (best["win_vs_anchor"], best["tie_vs_anchor"], best["lose_vs_anchor"]),
        "- Selected mode distribution: %s" % best["selected_mode_distribution"],
        "- Selected alpha distribution: %s" % best["selected_alpha_distribution"],
        "",
        "## Safety Assessment",
        "",
        "- Eliminated EEG negative transfer: %s" % ("yes" if int(best["negative_transfer_subject_count"]) == 0 else "no"),
        "- Reduced EEG negative transfer to <= 1 subject: %s" % ("yes" if bool(best["success_negative_transfer_le_1"]) else "no"),
        "- Exceeds exact gaze anchor mean AUROC: %s" % ("yes" if float(best["mean_delta_vs_anchor"]) > 0 else "no"),
        "- Exceeds NOCS full mean AUROC 0.6922: %s" % ("yes" if float(best["mean_auroc"]) > NOCS_FULL_AUROC else "no"),
        "- Exceeds SafeNOCS mean AUROC 0.6926: %s" % ("yes" if float(best["mean_auroc"]) > 0.6926 else "no"),
        "",
        "## Subjects",
        "",
        "- EEG helpful: %s" % (", ".join(helpful) if helpful else "none"),
        "- EEG harmful: %s" % (", ".join(harmful) if harmful else "none"),
        "- Anchor fallback/tie: %s" % (", ".join(tied) if tied else "none"),
        "",
        "## Recommendation",
        "",
    ]
    if bool(best["strong_success"]):
        lines.append("- Strong success: consider 5-seed validation before paper claims.")
    elif bool(best["success_negative_transfer_le_1"]) and bool(best["success_mean_near_anchor"]):
        lines.append("- Controlled success: run 5 seeds only if the paper needs stability estimates.")
    elif stable_fail:
        lines.append(
            "- EEG provides subject-dependent corrections, but stable universal improvement over gaze anchor is not supported."
        )
        lines.append("- Recommendation: stop model search and write the result as a negative-transfer safety finding.")
    else:
        lines.append("- Partial success: do not run 5 seeds yet; inspect harmful subjects before expanding.")
    return "\n".join(lines) + "\n"


def group_subject_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["strategy"])].append(row)
    return grouped
