"""Exploratory charts for the TBM stroke log."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data import LABELS, TARGET

NAVY, BLUE, RED = "#0f2a44", "#3d6d99", "#b2182b"
NUMERIC = ["advance_time_min", "excavation_m", "boring_energy_n_mm2", "rpm", "torque_knm", "thrust_kn",
           "speed_mm_min", "gripper_bar", "total_revolutions", "shield_lhs_bar", "shield_rhs_bar",
           "q_value_njc", TARGET]
sns.set_theme(style="whitegrid")


def _finish(ax, title):
    ax.set_title(title, fontsize=13, loc="left", fontweight="bold")
    sns.despine(ax=ax)
    ax.figure.tight_layout()
    return ax.figure


def correlations_with_target(df: pd.DataFrame) -> pd.DataFrame:
    c = df[NUMERIC].corr()[TARGET].drop(TARGET)
    return (pd.DataFrame({"column": c.index, "label": [LABELS[i] for i in c.index], "correlation": c.values})
            .sort_values("correlation").round(3).reset_index(drop=True))


def plot_penetration_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(df[TARGET], bins=40, color=BLUE, ax=ax)
    q25 = df[TARGET].quantile(0.25)
    ax.axvline(df[TARGET].median(), color=NAVY, linewidth=2, label=f"Median {df[TARGET].median():.1f}")
    ax.axvline(q25, color=RED, linestyle="--", linewidth=2, label=f"Slowest 25% below {q25:.1f}")
    ax.set_xlabel("Penetration (mm per cutterhead revolution)")
    ax.set_ylabel("Strokes")
    ax.legend(frameon=False)
    return _finish(ax, "Distribution of penetration per stroke")


def plot_penetration_check(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.scatter(df["speed_mm_min"] / df["rpm"], df[TARGET], s=10, alpha=0.5, color=BLUE)
    lim = [0, df[TARGET].max() + 2]
    ax.plot(lim, lim, color=RED, linestyle="--", linewidth=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Advance speed / RPM")
    ax.set_ylabel("Recorded penetration (mm/rev)")
    return _finish(ax, "Penetration is speed divided by RPM")


def plot_correlation_heatmap(df: pd.DataFrame):
    cols = ["thrust_kn", "torque_knm", "rpm", "gripper_bar", "shield_lhs_bar", "shield_rhs_bar", "q_value_njc",
            "speed_mm_min", "boring_energy_n_mm2", TARGET]
    corr = df[cols].corr()
    corr.index = corr.columns = [LABELS[c].split(" (")[0] for c in cols]
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, linewidths=0.5,
                annot_kws={"size": 8}, ax=ax)
    ax.set_title("Correlation between machine readings and penetration", fontsize=13, loc="left", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_drivers(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col in zip(axes, ["thrust_kn", "torque_knm", "rpm"]):
        sns.regplot(data=df, x=col, y=TARGET, scatter_kws={"s": 8, "alpha": 0.35, "color": BLUE},
                    line_kws={"color": RED}, ci=None, ax=ax)
        r = df[[col, TARGET]].corr().iloc[0, 1]
        ax.set_title(f"{LABELS[col]}  (r = {r:.2f})", loc="left", fontweight="bold", fontsize=11)
        ax.set_xlabel(LABELS[col])
        ax.set_ylabel("Penetration (mm/rev)" if col == "thrust_kn" else "")
        sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_by_section(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    sns.boxplot(data=df, x="section", y=TARGET, color="#cfe2f3", fliersize=2, width=0.6, ax=ax)
    ax.set_xlabel("Section of the log")
    ax.set_ylabel("Penetration (mm/rev)")
    return _finish(ax, "Penetration by section")


def plot_rock_quality(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    sns.boxplot(data=df, x="q_value_njc", y=TARGET, color="#cfe2f3", width=0.5, fliersize=2, ax=ax)
    counts = df["q_value_njc"].value_counts().sort_index()
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([f"Q = {q}\n({n} strokes)" for q, n in counts.items()])
    ax.set_xlabel("")
    ax.set_ylabel("Penetration (mm/rev)")
    return _finish(ax, "Penetration by rock mass Q-value")
