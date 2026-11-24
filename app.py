"""Streamlit app: TBM penetration analysis and next-stroke prediction.

Run locally:
    streamlit run app.py
"""

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

from src.analysis import (
    correlations_with_target,
    plot_by_section,
    plot_correlation_heatmap,
    plot_drivers,
    plot_penetration_check,
    plot_penetration_distribution,
    plot_rock_quality,
)
from src.data import TARGET, clean, load_raw, quality_report, section_summary
from src.model import FEATURES, final_model, low_penetration_threshold, modeling_rows

# Out-of-section results from the notebook (notebooks/tbm_penetration_prediction.ipynb).
RESULTS = pd.DataFrame(
    [
        ["Always predict the average", "none", 1.04, -0.02],
        ["Repeat the previous stroke", "previous stroke", 0.84, 0.25],
        ["Gradient Boosting", "machine readings and rock quality", 0.86, 0.32],
        ["Ridge Regression", "machine readings, rock quality and history", 0.75, 0.44],
        ["Random Forest", "machine readings, rock quality and history", 0.72, 0.47],
    ],
    columns=["model", "inputs", "MAE (mm/rev)", "R2"],
)
LEAKAGE = pd.DataFrame(
    [
        ["Machine readings, rock quality and history", 0.72, 0.47],
        ["... plus advance speed (penetration = speed / RPM)", 0.26, 0.89],
        ["... plus boring energy (= torque x RPM / speed)", 0.52, 0.68],
    ],
    columns=["inputs", "MAE (mm/rev)", "R2"],
)

st.set_page_config(page_title="TBM Penetration Prediction", layout="wide")


@st.cache_data
def get_data():
    raw = load_raw()
    df, removed = clean(raw)
    return raw, df, removed


@st.cache_resource
def get_models(_df):
    d = modeling_rows(_df)
    reg = final_model().fit(d[FEATURES], d[TARGET])
    threshold = low_penetration_threshold(_df)
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced", n_jobs=-1,
                                 random_state=42).fit(d[FEATURES], (d[TARGET] < threshold).astype(int))
    return reg, clf, threshold


raw, df, removed = get_data()
reg, clf, threshold = get_models(df)

st.title("TBM Penetration Prediction")
st.write(
    f"Stroke-by-stroke analysis of a tunnel boring machine log ({len(df):,} strokes, 2016) and a model that "
    "predicts the penetration of the next stroke from machine readings, rock quality and recent strokes."
)

tab_predict, tab_explore, tab_model, tab_quality = st.tabs(
    ["Predict next stroke", "Explore the log", "Model", "Data quality"]
)

# ---------------------------------------------------------------- prediction
with tab_predict:
    st.subheader("Predict the penetration of the next stroke")
    med = df.median(numeric_only=True)
    c1, c2, c3 = st.columns(3)
    thrust = c1.slider("Thrust force (kN)", 1000, 10000, int(round(med["thrust_kn"], -2)), step=100)
    torque = c1.slider("Torque (kN-m)", 100, 4000, int(round(med["torque_knm"], -2)), step=100)
    rpm = c1.slider("Cutterhead RPM", 2.0, 5.5, float(round(med["rpm"], 1)), step=0.1)
    gripper = c2.slider("Gripper pressure (bar)", 160, 300, int(med["gripper_bar"]))
    lhs = c2.slider("Front shield pressure LHS (bar)", 10, 100, int(med["shield_lhs_bar"]))
    rhs = c2.slider("Front shield pressure RHS (bar)", 10, 100, int(med["shield_rhs_bar"]))
    q_value = c3.radio("Rock mass Q-value (NJC)", [3, 4], index=1, horizontal=True)
    prev = c3.number_input("Penetration of the previous stroke (mm/rev)", 3.0, 16.0, 12.0, step=0.1)
    last5 = c3.number_input("Average of the last 5 strokes (mm/rev)", 3.0, 16.0, 12.0, step=0.1)

    row = pd.DataFrame([{
        "thrust_kn": thrust, "torque_knm": torque, "rpm": rpm, "gripper_bar": gripper,
        "shield_lhs_bar": lhs, "shield_rhs_bar": rhs, "q_value_njc": q_value,
        "prev_penetration": prev, "mean_penetration_last_5": last5,
    }])[FEATURES]
    pen = float(reg.predict(row)[0])
    slow = float(clf.predict_proba(row)[0, 1])

    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted penetration", f"{pen:.1f} mm/rev")
    m2.metric("Expected advance speed", f"{pen * rpm:.0f} mm/min")
    m3.metric(f"Chance of a slow stroke (below {threshold:.1f} mm/rev)", f"{slow * 100:.0f}%")
    st.caption(
        "Expected advance speed is predicted penetration times RPM. The model's average error on sections it "
        "never saw is 0.72 mm/rev, so treat the prediction as plus or minus about 0.7 mm/rev."
    )

# ---------------------------------------------------------------- explore
with tab_explore:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Strokes", f"{len(df):,}")
    k2.metric("Median penetration", f"{df[TARGET].median():.1f} mm/rev")
    k3.metric("Median thrust", f"{df['thrust_kn'].median():,.0f} kN")
    k4.metric("Sections", df["section"].nunique())
    left, right = st.columns(2)
    left.pyplot(plot_penetration_distribution(df))
    right.pyplot(plot_rock_quality(df))
    st.pyplot(plot_drivers(df))
    st.caption(
        "Higher thrust goes with lower penetration. Operators push harder in harder rock, so thrust here mostly "
        "reflects how hard the ground is, not a setting that slows the machine down."
    )
    left, right = st.columns([3, 2])
    left.pyplot(plot_correlation_heatmap(df))
    right.dataframe(correlations_with_target(df)[["label", "correlation"]], hide_index=True, width="stretch")
    st.pyplot(plot_by_section(df))
    st.dataframe(section_summary(df), hide_index=True, width="stretch")

# ---------------------------------------------------------------- model
with tab_model:
    st.markdown(
        "Every score below comes from **leaving whole sections of the log out** (GroupKFold): a section is "
        "predicted by a model trained only on the other sections. Neighbouring strokes are very similar, so a "
        "random split would let the model copy answers from the stroke next to it and overstate its accuracy."
    )
    st.dataframe(RESULTS, hide_index=True, width="stretch")
    st.markdown(
        "- The Random Forest with history is the final model: R2 0.47, average error 0.72 mm/rev, and 75% of "
        "strokes predicted within 1 mm/rev.\n"
        "- It relies most on the previous stroke's penetration, then thrust, RPM and torque.\n"
        "- Flagging slow strokes (the slowest 25%, below 11.3 mm/rev): ROC AUC 0.83, catching 67% of slow "
        "strokes with 16% false alarms."
    )
    st.subheader("Why advance speed and boring energy are not inputs")
    left, right = st.columns([2, 3])
    left.pyplot(plot_penetration_check(df))
    right.write(
        "Penetration is recorded as advance speed divided by RPM, and boring energy is torque times RPM divided "
        "by speed. Either column gives the model the answer. They look like strong predictors, but they are only "
        "known after the stroke is finished."
    )
    right.dataframe(LEAKAGE, hide_index=True, width="stretch")
    right.write(
        "The first version of this project also built its labels with a fixed pressure threshold that put every "
        "stroke in the same class, so its 100% accuracy meant nothing. This version uses the real penetration values."
    )

# ---------------------------------------------------------------- data quality
with tab_quality:
    st.dataframe(quality_report(raw).astype(str).rename("value"), width="stretch")
    st.markdown("**Removed strokes**")
    st.dataframe(removed, hide_index=True, width="stretch")
    st.markdown(
        "- The sheet has a blank row, a header row and a units row, which are read separately.\n"
        "- The serial number restarts 10 times, splitting the log into 11 sections.\n"
        "- Stroke numbers repeat across sections with different readings, so they are not treated as duplicates.\n"
        "- Some sections have day and month swapped in the date, so dates are not used by the model."
    )
    st.download_button("Download the cleaned data", df.to_csv(index=False), file_name="tbm_strokes_clean.csv")
