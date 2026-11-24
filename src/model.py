"""Penetration models for the TBM stroke log.

Penetration (mm per cutterhead revolution) equals advance speed divided by RPM, and
boring energy is torque x RPM / speed. Both therefore contain the answer and are not
used as inputs. Models are validated by leaving out whole sections of the log
(GroupKFold), because neighbouring strokes are very similar and a random split would
let the model copy the answer from the stroke next to it.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, mean_absolute_error, r2_score,
                             roc_auc_score)
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import LABELS, TARGET

MACHINE = ["thrust_kn", "torque_knm", "rpm", "gripper_bar", "shield_lhs_bar", "shield_rhs_bar", "q_value_njc"]
HISTORY = ["prev_penetration", "mean_penetration_last_5"]
FEATURES = MACHINE + HISTORY
LEAKY = {"speed_mm_min": "Advance speed (penetration = speed / RPM)",
         "boring_energy_n_mm2": "Boring energy (= torque x RPM / speed)"}
RANDOM_STATE = 42
NAVY, BLUE, RED = "#0f2a44", "#3d6d99", "#b2182b"
sns.set_theme(style="whitegrid")


def modeling_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Strokes that have a previous stroke in the same section."""
    return df.dropna(subset=HISTORY).reset_index(drop=True)


def regressors() -> dict:
    return {
        "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=-1,
                                               random_state=RANDOM_STATE),
        "Gradient Boosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=4,
                                                           random_state=RANDOM_STATE),
    }


def final_model():
    return regressors()["Random Forest"]


def _oof(model, X, y, groups):
    return cross_val_predict(model, X, y, cv=GroupKFold(n_splits=5), groups=groups)


def _scores(y, p) -> dict:
    return {"R2": r2_score(y, p), "MAE_mm_rev": mean_absolute_error(y, p)}


def compare_models(df: pd.DataFrame) -> pd.DataFrame:
    """Out-of-section scores for simple rules and for each model on two feature sets."""
    d = modeling_rows(df)
    y, g = d[TARGET], d["section"]
    rows = [
        {"inputs": "none", "model": "Always predict the average", **_scores(y, _oof(DummyRegressor(), d[MACHINE], y, g))},
        {"inputs": "previous stroke", "model": "Repeat the previous stroke", **_scores(y, d["prev_penetration"])},
    ]
    for inputs, cols in [("machine readings and rock quality", MACHINE), ("machine readings, rock quality and history", FEATURES)]:
        for name, model in regressors().items():
            rows.append({"inputs": inputs, "model": name, **_scores(y, _oof(model, d[cols], y, g))})
    return pd.DataFrame(rows).round({"R2": 3, "MAE_mm_rev": 2})


def validation_check(df: pd.DataFrame) -> pd.DataFrame:
    """Random split vs section split for the final model."""
    d = modeling_rows(df)
    y = d[TARGET]
    random_p = cross_val_predict(final_model(), d[FEATURES], y, cv=KFold(5, shuffle=True, random_state=RANDOM_STATE))
    group_p = _oof(final_model(), d[FEATURES], y, d["section"])
    return pd.DataFrame([
        {"validation": "Random 5-fold split", **_scores(y, random_p)},
        {"validation": "Leave whole sections out (GroupKFold)", **_scores(y, group_p)},
    ]).round({"R2": 3, "MAE_mm_rev": 2})


def leakage_demo(df: pd.DataFrame) -> pd.DataFrame:
    d = modeling_rows(df)
    y, g = d[TARGET], d["section"]
    rows = [{"inputs": "Machine readings, rock quality and history", **_scores(y, _oof(final_model(), d[FEATURES], y, g))}]
    for col, label in LEAKY.items():
        rows.append({"inputs": f"... plus {label}", **_scores(y, _oof(final_model(), d[FEATURES + [col]], y, g))})
    return pd.DataFrame(rows).round({"R2": 3, "MAE_mm_rev": 2})


def out_of_section_predictions(df: pd.DataFrame) -> pd.DataFrame:
    d = modeling_rows(df)
    out = d[["section", "sr", "stroke_no", TARGET]].copy()
    out["predicted"] = _oof(final_model(), d[FEATURES], d[TARGET], d["section"]).round(2)
    out["error"] = (out["predicted"] - out[TARGET]).round(2)
    return out


def feature_importance(df: pd.DataFrame) -> pd.DataFrame:
    """Permutation importance, each section scored by a model that never saw it."""
    d = modeling_rows(df)
    results = []
    for train_idx, test_idx in GroupKFold(n_splits=5).split(d, groups=d["section"]):
        model = final_model().fit(d.loc[train_idx, FEATURES], d.loc[train_idx, TARGET])
        r = permutation_importance(model, d.loc[test_idx, FEATURES], d.loc[test_idx, TARGET], n_repeats=10,
                                   random_state=RANDOM_STATE, scoring="neg_mean_absolute_error")
        results.append(r.importances_mean)
    imp = pd.DataFrame({"feature": FEATURES, "MAE_increase": np.mean(results, axis=0)})
    imp["label"] = imp["feature"].map(lambda f: LABELS.get(f, f.replace("_", " ").capitalize()))
    return imp.sort_values("MAE_increase", ascending=False).round(3).reset_index(drop=True)


# ---------------------------------------------------------------- classification


def low_penetration_threshold(df: pd.DataFrame) -> float:
    return float(df[TARGET].quantile(0.25))


def classify_low_penetration(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, float]:
    """Flag strokes in the slowest quarter, out of section."""
    d = modeling_rows(df)
    threshold = low_penetration_threshold(df)
    y = (d[TARGET] < threshold).astype(int)
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                                n_jobs=-1, random_state=RANDOM_STATE),
    }
    rows, best_cm = [], None
    for name, model in models.items():
        proba = cross_val_predict(model, d[FEATURES], y, cv=GroupKFold(5), groups=d["section"],
                                  method="predict_proba")[:, 1]
        pred = (proba >= 0.5).astype(int)
        cm = confusion_matrix(y, pred)
        rows.append({"model": name, "ROC_AUC": roc_auc_score(y, proba),
                     "balanced_accuracy": balanced_accuracy_score(y, pred),
                     "slow_strokes_caught_%": cm[1, 1] / cm[1].sum() * 100,
                     "false_alarms_%": cm[0, 1] / cm[0].sum() * 100})
        if name == "Random Forest":
            best_cm = cm
    return pd.DataFrame(rows).round(3), best_cm, threshold


# ---------------------------------------------------------------- charts


def _finish(ax, title):
    ax.set_title(title, fontsize=13, loc="left", fontweight="bold")
    sns.despine(ax=ax)
    ax.figure.tight_layout()
    return ax.figure


def plot_model_comparison(comparison: pd.DataFrame):
    c = comparison.copy()
    c["label"] = c["model"] + np.where(c["inputs"].str.contains("history"), "  (+ history)",
                                       np.where(c["inputs"].str.startswith("machine"), "  (machine only)", ""))
    colors = [RED if i in ("none", "previous stroke") else (NAVY if "history" in i else BLUE) for i in c["inputs"]]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.barh(c["label"], c["MAE_mm_rev"], color=colors)
    for i, (v, r2) in enumerate(zip(c["MAE_mm_rev"], c["R2"])):
        ax.text(v + 0.01, i, f"MAE {v:.2f}  |  R2 {r2:.2f}", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, c["MAE_mm_rev"].max() * 1.45)
    ax.set_xlabel("Mean absolute error on unseen sections (mm/rev, lower is better)")
    return _finish(ax, "Model comparison on sections the model never saw")


def plot_actual_vs_predicted(preds: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    ax.scatter(preds[TARGET], preds["predicted"], s=12, alpha=0.5, color=BLUE)
    lim = [preds[[TARGET, "predicted"]].min().min() - 0.5, preds[[TARGET, "predicted"]].max().max() + 0.5]
    ax.plot(lim, lim, color="#555", linestyle="--", linewidth=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Actual penetration (mm/rev)")
    ax.set_ylabel("Predicted penetration (mm/rev)")
    return _finish(ax, "Random Forest: actual vs predicted")


def plot_section_trace(df: pd.DataFrame, preds: pd.DataFrame, section: int):
    p = preds[preds["section"] == section]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(p["sr"], p[TARGET], color=NAVY, linewidth=1.4, label="Actual")
    ax.plot(p["sr"], p["predicted"], color=RED, linewidth=1.4, alpha=0.85, label="Predicted")
    ax.set_xlabel("Stroke in section")
    ax.set_ylabel("Penetration (mm/rev)")
    ax.legend(frameon=False, ncol=2)
    return _finish(ax, f"Section {section}: actual vs predicted, stroke by stroke")


def plot_feature_importance(imp: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(imp["label"], imp["MAE_increase"], color=NAVY)
    ax.invert_yaxis()
    ax.set_xlabel("Increase in MAE when the input is shuffled (mm/rev)")
    return _finish(ax, "What the model relies on")
