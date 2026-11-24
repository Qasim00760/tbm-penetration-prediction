"""Loading and cleaning the TBM stroke log."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "tbm_strokes.xlsx"
TARGET = "penetration_mm_rev"

# The sheet has a blank first row, a header row and a units row.
COLUMNS = {
    "Sr": "sr",
    "Stroke No.": "stroke_no",
    "Date": "date",
    "Advance Time": "advance_time_min",
    "Excavation per Stroke": "excavation_m",
    "Boring Energy": "boring_energy_n_mm2",
    "RPM": "rpm",
    "Torque": "torque_knm",
    "Thrust Force": "thrust_kn",
    "Speed": "speed_mm_min",
    "Gripper Pressure": "gripper_bar",
    "Total Revolution": "total_revolutions",
    "Front Shield Pressure LHS": "shield_lhs_bar",
    "Front Shield Pressure rHS": "shield_rhs_bar",
    "Q-Value": "q_value_njc",
    "Q-Value based": "q_class_from_energy",
    "Penetration": TARGET,
}

LABELS = {
    "advance_time_min": "Advance time (min)",
    "excavation_m": "Excavation per stroke (m)",
    "boring_energy_n_mm2": "Boring energy (N/mm2)",
    "rpm": "Cutterhead RPM",
    "torque_knm": "Torque (kN-m)",
    "thrust_kn": "Thrust force (kN)",
    "speed_mm_min": "Advance speed (mm/min)",
    "gripper_bar": "Gripper pressure (bar)",
    "total_revolutions": "Total revolutions",
    "shield_lhs_bar": "Front shield pressure LHS (bar)",
    "shield_rhs_bar": "Front shield pressure RHS (bar)",
    "q_value_njc": "Rock mass Q-value (NJC)",
    "q_class_from_energy": "Q class from boring energy",
    TARGET: "Penetration (mm/rev)",
}


def load_raw(path=RAW) -> pd.DataFrame:
    header = pd.read_excel(path, header=None, nrows=3)
    names = [str(v).strip() for v in header.iloc[1]]
    df = pd.read_excel(path, header=None, skiprows=3)
    df.columns = [COLUMNS[n] for n in names]
    return df


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return clean strokes and a table of the rows that were removed and why."""
    df = raw.copy()
    # The serial number restarts for each section of the log, so it marks the sections.
    df["section"] = (df["sr"].diff() < 0).cumsum() + 1
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    rules = {
        "gripper pressure above 400 bar": df["gripper_bar"] > 400,
        "front shield pressure RHS below 5 or above 120 bar": (df["shield_rhs_bar"] < 5) | (df["shield_rhs_bar"] > 120),
        "excavation per stroke 0 or less, or above 2.5 m": (df["excavation_m"] <= 0) | (df["excavation_m"] > 2.5),
        "advance time of 0 minutes": df["advance_time_min"] == 0,
        "penetration above 25 mm/rev (three times the next highest)": df[TARGET] > 25,
    }
    reasons = pd.Series("", index=df.index)
    for reason, mask in rules.items():
        reasons[mask & (reasons == "")] = reason
    removed = df.loc[reasons != "", ["section", "sr", "stroke_no"]].assign(reason=reasons[reasons != ""])
    df = df[reasons == ""].copy()

    # History features: what the operator already knows before the stroke starts.
    by_section = df.groupby("section")[TARGET]
    df["prev_penetration"] = by_section.shift(1)
    df["mean_penetration_last_5"] = by_section.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    return df.reset_index(drop=True), removed.reset_index(drop=True)


def load_clean(path=RAW) -> pd.DataFrame:
    return clean(load_raw(path))[0]


def quality_report(raw: pd.DataFrame) -> pd.Series:
    sections = (raw["sr"].diff() < 0).cumsum() + 1
    dates = pd.to_datetime(raw["date"], errors="coerce")
    return pd.Series({
        "strokes in the sheet": len(raw),
        "missing values": int(raw.isna().sum().sum()),
        "sections (serial number restarts)": int(sections.max()),
        "stroke numbers that appear more than once": int(raw["stroke_no"].duplicated().sum()),
        "fully duplicated rows": int(raw.drop(columns="sr").duplicated().sum()),
        "dates that could not be read": int(dates.isna().sum()),
        "rows where penetration = speed / RPM (within 0.6)":
            int(((raw["speed_mm_min"] / raw["rpm"] - raw[TARGET]).abs() < 0.6).sum()),
    })


def section_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("section")
    return pd.DataFrame({
        "strokes": g.size(),
        "first_stroke": g["stroke_no"].min(),
        "last_stroke": g["stroke_no"].max(),
        "mean_penetration": g[TARGET].mean().round(2),
        "mean_thrust_kn": g["thrust_kn"].mean().round(0),
        "share_q4_%": (g["q_value_njc"].apply(lambda s: (s == 4).mean()) * 100).round(0),
    }).reset_index()
