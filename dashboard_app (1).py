"""
High Resource Utilizer Prediction — Stakeholder Dashboard
===========================================================
Deployment model: Gradient Boosting (AUROC 0.852, Recall 81.4%)

Run with:
    streamlit run app.py

Expects the following CSVs from the notebook's OUT_DIR (default: ./outputs_classification/).
If a file is missing, that section shows a clear "data not found" notice rather than crashing,
so the app is always safe to demo even with a partial output folder.

    patient_risk_scores.csv          subject_id, hadm_id, predicted_probability,
                                      flagged_high_risk, actual_outcome
    model_comparison_summary.csv     index=model name; AUROC, Precision, Recall, F1
    feature_importance.csv           index=feature; single importance column
    shap_importance.csv              index=feature; single mean|SHAP| column
    calibration_summary_all_models.csv  index=model name; brier_score, n_at_extremes
    fairness_audit_by_insurance.csv  insurance, n, actual_high_risk, fnr
    cohort_with_features.csv         full cohort (used for population-level insights)
    eval_13_confusion_matrix.png     (optional, displayed directly if present)
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------------------------------------------------------------
# Page config & styling
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="High-Risk Patient Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#0B5FA5"
HIGH = "#C0392B"
MED = "#E1A200"
LOW = "#1E8449"
NEUTRAL_BG = "#F7F9FB"

st.markdown(f"""
<style>
    .main {{ background-color: {NEUTRAL_BG}; }}
    div[data-testid="stMetric"] {{
        background-color: white;
        border: 1px solid #E3E8EE;
        border-radius: 10px;
        padding: 14px 16px 10px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }}
    div[data-testid="stMetricValue"] {{ font-size: 1.8rem; color: {PRIMARY}; }}
    .badge-high {{ background:{HIGH}; color:white; padding:2px 10px; border-radius:12px; font-size:0.8rem; font-weight:600; }}
    .badge-med  {{ background:{MED};  color:white; padding:2px 10px; border-radius:12px; font-size:0.8rem; font-weight:600; }}
    .badge-low  {{ background:{LOW};  color:white; padding:2px 10px; border-radius:12px; font-size:0.8rem; font-weight:600; }}
    .model-chip {{
        display:inline-block; background:#EAF2FB; color:{PRIMARY}; border:1px solid #CFE1F5;
        border-radius:8px; padding:3px 12px; font-size:0.85rem; font-weight:600; margin-bottom:6px;
    }}
    h1, h2, h3 {{ color:#1B2733; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
    .stTabs [data-baseweb="tab"] {{
        background-color:white; border-radius:8px 8px 0 0; padding:8px 18px; border:1px solid #E3E8EE;
    }}
</style>
""", unsafe_allow_html=True)

DATA_DIR = st.sidebar.text_input("Outputs folder", value="./outputs_classification")
DEPLOYED_MODEL = "Gradient Boosting"


# --------------------------------------------------------------------------------------
# Data loading — cached, resilient to missing files
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(data_dir, name, index_col=None):
    path = os.path.join(data_dir, name)
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, index_col=index_col)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_all(data_dir):
    return {
        "risk": load_csv(data_dir, "patient_risk_scores.csv"),
        "comparison": load_csv(data_dir, "model_comparison_summary.csv", index_col=0),
        "feat_imp": load_csv(data_dir, "feature_importance.csv", index_col=0),
        "shap_imp": load_csv(data_dir, "shap_importance.csv", index_col=0),
        "calibration": load_csv(data_dir, "calibration_summary_all_models.csv", index_col=0),
        "fairness": load_csv(data_dir, "fairness_audit_by_insurance.csv"),
        "cohort": load_csv(data_dir, "cohort_with_features.csv"),
    }


data = load_all(DATA_DIR)
risk_df = data["risk"]


def missing_notice(label):
    st.info(
        f"**{label} data not found.** Place the notebook's `{label}` CSV in "
        f"`{DATA_DIR}` and reload — this section will populate automatically. "
        f"The rest of the dashboard is unaffected."
    )


def risk_tier(p):
    if p >= 0.70:
        return "High"
    elif p >= 0.40:
        return "Medium"
    return "Low"


def risk_badge(tier):
    cls = {"High": "badge-high", "Medium": "badge-med", "Low": "badge-low"}[tier]
    return f'<span class="{cls}">{tier}</span>'


if risk_df is not None and "predicted_probability" in risk_df.columns:
    risk_df = risk_df.copy()
    risk_df["risk_tier"] = risk_df["predicted_probability"].apply(risk_tier)

# --------------------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------------------
left, right = st.columns([3, 1])
with left:
    st.title("🏥 High Resource Utilizer Prediction")
    st.markdown(
        f'<span class="model-chip">Deployed model: {DEPLOYED_MODEL}</span> '
        f'<span class="model-chip">Data: MIMIC-IV, 545,848 hospitalizations</span> '
        f'<span class="model-chip">Predicts at admission, before length of stay is known</span>',
        unsafe_allow_html=True,
    )
with right:
    if risk_df is not None:
        st.metric("Patients in view", f"{len(risk_df):,}")

st.divider()

# --------------------------------------------------------------------------------------
# Tabs — one per stakeholder audience
# --------------------------------------------------------------------------------------
tab_exec, tab_care, tab_ops, tab_list, tab_model = st.tabs([
    "📊 Executive Summary",
    "🩺 Care Team Priority Outreach",
    "🏢 Operations & Capacity Planning",
    "📋 Full Patient Risk List",
    "🔎 Model Transparency & Fairness",
])

# ======================================================================================
# TAB 1 — EXECUTIVE SUMMARY
# ======================================================================================
with tab_exec:
    st.subheader("Single-screen overview for leadership review")

    if risk_df is None:
        missing_notice("patient_risk_scores.csv")
    else:
        n_total = len(risk_df)
        n_high = (risk_df["risk_tier"] == "High").sum()
        n_med = (risk_df["risk_tier"] == "Medium").sum()
        n_low = (risk_df["risk_tier"] == "Low").sum()
        recall = data["comparison"].loc[DEPLOYED_MODEL, "Recall"] if data["comparison"] is not None and DEPLOYED_MODEL in data["comparison"].index else None

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Admissions analyzed", f"{n_total:,}")
        c2.metric("Flagged high-risk", f"{n_high:,}", f"{n_high/n_total*100:.0f}% of total")
        if recall is not None:
            c3.metric("Model reliability", f"{recall*100:.0f}%",
                      help="Recall: of every 10 patients who truly become high-resource, this many are correctly flagged at admission.")
        c4.metric("Care teams supported", "3", help="Care Management, Operations, Executive")

        if recall is not None:
            st.success(
                f"**In plain language:** this model correctly identifies **{recall*10:.0f} out of every 10 patients** "
                f"who truly turn out to be high-risk — using only information available at the point of admission, "
                f"at no additional data-collection cost to the hospital."
            )

        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.markdown("##### Risk mix across admitted patients")
            donut = go.Figure(data=[go.Pie(
                labels=["Low risk", "Medium risk", "High risk"],
                values=[n_low, n_med, n_high],
                hole=0.55,
                marker=dict(colors=[LOW, MED, HIGH]),
            )])
            donut.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320,
                                 legend=dict(orientation="h", y=-0.1))
            st.plotly_chart(donut, use_container_width=True)
        with col_b:
            st.markdown("##### Predicted risk distribution")
            fig = px.histogram(
                risk_df, x="predicted_probability", color="actual_outcome" if "actual_outcome" in risk_df.columns else None,
                nbins=30, color_discrete_map={"High-utilizer": HIGH, "Not high-utilizer": PRIMARY},
                labels={"predicted_probability": "Predicted probability of high resource use"},
            )
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320, legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)

        if data["comparison"] is not None:
            st.markdown("##### Model comparison at a glance")
            comp = data["comparison"].copy().sort_values("AUROC", ascending=False)
            st.dataframe(
                comp.style.format("{:.3f}").apply(
                    lambda s: ["background-color:#EAF6EC" if s.name == DEPLOYED_MODEL else "" for _ in s], axis=1
                ),
                use_container_width=True,
            )
            st.caption(f"**{DEPLOYED_MODEL}** (highlighted) is the deployed model — selected for leading on both AUROC and recall.")

# ======================================================================================
# TAB 2 — CARE TEAM PRIORITY OUTREACH
# ======================================================================================
with tab_care:
    st.subheader("Ranked, plain-language outreach list for care management staff")

    if risk_df is None:
        missing_notice("patient_risk_scores.csv")
    else:
        n_high = (risk_df["risk_tier"] == "High").sum()
        n_med = (risk_df["risk_tier"] == "Medium").sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("Total patients in list", f"{len(risk_df):,}")
        c2.metric("🔴 High priority", f"{n_high:,}")
        c3.metric("🟡 Medium priority", f"{n_med:,}", help="Worth monitoring; lower urgency than High")

        st.markdown("##### Priority outreach queue")
        tier_filter = st.multiselect("Show tiers", ["High", "Medium", "Low"], default=["High", "Medium"])
        n_show = st.slider("Patients to display", 5, 100, 25, step=5)

        top_feats = None
        if data["shap_imp"] is not None:
            top_feats = data["shap_imp"].iloc[:, 0].sort_values(ascending=False).head(5).index.tolist()

        def plain_reason(row):
            # Simple, defensible plain-language reason from available cohort features if present
            reasons = []
            cohort = data["cohort"]
            if cohort is not None and "hadm_id" in cohort.columns and "hadm_id" in risk_df.columns:
                match = cohort[cohort["hadm_id"] == row.get("hadm_id")]
                if not match.empty:
                    m = match.iloc[0]
                    if m.get("n_diagnoses", 0) and m.get("n_diagnoses", 0) >= cohort["n_diagnoses"].median():
                        reasons.append("Multiple ongoing health conditions")
                    if m.get("icu_stay_count_24h", 0) and m.get("icu_stay_count_24h", 0) > 0:
                        reasons.append("ICU stay during this admission")
            return " · ".join(reasons) if reasons else "See predicted probability and top model drivers"

        view = risk_df[risk_df["risk_tier"].isin(tier_filter)].sort_values("predicted_probability", ascending=False).head(n_show).copy()
        view["Risk"] = view["risk_tier"].apply(risk_badge)
        view["Probability"] = (view["predicted_probability"] * 100).round(1).astype(str) + "%"
        view["Reason"] = view.apply(plain_reason, axis=1)

        display_cols = ["subject_id", "hadm_id", "Risk", "Probability", "Reason"]
        display_cols = [c for c in display_cols if c in view.columns]
        st.write(
            view[display_cols].to_html(escape=False, index=False),
            unsafe_allow_html=True,
        )

        if top_feats:
            st.caption(f"Top model drivers overall: {', '.join(top_feats)}. See **Model Transparency** tab for full detail.")

# ======================================================================================
# TAB 3 — OPERATIONS & CAPACITY PLANNING
# ======================================================================================
with tab_ops:
    st.subheader("Workload and capacity view for hospital operations & finance")

    if risk_df is None:
        missing_notice("patient_risk_scores.csv")
    else:
        n_high = (risk_df["risk_tier"] == "High").sum()
        n_med = (risk_df["risk_tier"] == "Medium").sum()
        est_calls = n_high

        c1, c2, c3 = st.columns(3)
        c1.metric("Estimated outreach workload", f"{est_calls:,} calls", help="One call per High-priority patient")
        c2.metric("Medium-priority for monitoring", f"{n_med:,}")
        c3.metric("Total hospitalizations modeled", f"{len(risk_df):,}")

        if data["cohort"] is not None:
            st.markdown("##### Repeat hospitalization patterns")
            cohort = data["cohort"]
            if "subject_id" in cohort.columns:
                counts = cohort.groupby("subject_id").size()
                repeat_share = (counts >= 2).mean() * 100
                cc1, cc2, cc3 = st.columns(3)
                cc1.metric("Patients with 2+ admissions", f"{repeat_share:.1f}%")
                cc2.metric("Avg. admissions per patient", f"{counts.mean():.2f}")
                cc3.metric("Max admissions, single patient", f"{counts.max():,}")

        if data["comparison"] is not None:
            st.markdown("##### Train vs test stability (overfitting check)")
            st.caption("A near-zero gap between train and test performance means results should hold up on new admissions.")

        st.markdown("##### Priority mix by tier")
        tier_counts = risk_df["risk_tier"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0)
        bar = px.bar(x=tier_counts.index, y=tier_counts.values,
                     color=tier_counts.index,
                     color_discrete_map={"High": HIGH, "Medium": MED, "Low": LOW},
                     labels={"x": "Risk tier", "y": "Patients"})
        bar.update_layout(showlegend=False, height=320, margin=dict(t=10, b=10))
        st.plotly_chart(bar, use_container_width=True)

# ======================================================================================
# TAB 4 — FULL PATIENT RISK LIST
# ======================================================================================
with tab_list:
    st.subheader("Full, searchable patient risk list")

    if risk_df is None:
        missing_notice("patient_risk_scores.csv")
    else:
        search_id = st.text_input("Search by subject_id or hadm_id")
        min_prob, max_prob = st.slider("Predicted probability range", 0.0, 1.0, (0.0, 1.0), step=0.01)

        filtered = risk_df[
            (risk_df["predicted_probability"] >= min_prob) & (risk_df["predicted_probability"] <= max_prob)
        ]
        if search_id:
            mask = pd.Series(False, index=filtered.index)
            for col in ["subject_id", "hadm_id"]:
                if col in filtered.columns:
                    mask |= filtered[col].astype(str).str.contains(search_id, na=False)
            filtered = filtered[mask]

        st.caption(f"Showing {len(filtered):,} of {len(risk_df):,} patients")
        st.dataframe(
            filtered.sort_values("predicted_probability", ascending=False),
            use_container_width=True,
            height=500,
        )

        csv_bytes = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download filtered list as CSV", csv_bytes, "filtered_patient_risk_list.csv", "text/csv")

# ======================================================================================
# TAB 5 — MODEL TRANSPARENCY & FAIRNESS
# ======================================================================================
with tab_model:
    st.subheader("How the model works, and whether it's fair")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Feature importance")
        if data["feat_imp"] is not None:
            fi = data["feat_imp"].iloc[:, 0].sort_values(ascending=True).tail(12)
            fig = px.bar(x=fi.values, y=fi.index, orientation="h",
                         labels={"x": "Importance", "y": ""}, color_discrete_sequence=[PRIMARY])
            fig.update_layout(height=380, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            missing_notice("feature_importance.csv")

    with col2:
        st.markdown("##### SHAP importance (with direction, see report Ch.12)")
        if data["shap_imp"] is not None:
            si = data["shap_imp"].iloc[:, 0].sort_values(ascending=True).tail(12)
            fig = px.bar(x=si.values, y=si.index, orientation="h",
                         labels={"x": "Mean |SHAP value|", "y": ""}, color_discrete_sequence=["#7A4FAE"])
            fig.update_layout(height=380, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            missing_notice("shap_importance.csv")

    st.divider()

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("##### Calibration — is a predicted 70% really 70%?")
        if data["calibration"] is not None:
            cal = data["calibration"].copy().sort_values("brier_score")
            st.dataframe(cal.style.format({"brier_score": "{:.4f}"}), use_container_width=True)
            st.caption("Lower Brier score is better. 0 patients at 0%/100% for the deployed model confirms no unrealistic saturation.")
        else:
            missing_notice("calibration_summary_all_models.csv")

    with col4:
        st.markdown("##### Confusion matrix")
        cm_path = os.path.join(DATA_DIR, "eval_13_confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, use_container_width=True)
        else:
            missing_notice("eval_13_confusion_matrix.png")

    st.divider()

    st.markdown("##### Fairness audit — false-negative rate by insurance category")
    st.caption("A higher false-negative rate for a group means the model is quietly missing more true high-risk patients in that group.")
    if data["fairness"] is not None:
        fair = data["fairness"].copy()
        fnr_col = [c for c in fair.columns if "fnr" in c.lower()]
        n_col = [c for c in fair.columns if c.lower() == "n"]
        if fnr_col:
            fair = fair.sort_values(fnr_col[0], ascending=False)
            fig = px.bar(fair, x=fair.columns[0], y=fnr_col[0],
                         color=fnr_col[0], color_continuous_scale=["#1E8449", "#E1A200", "#C0392B"],
                         labels={fnr_col[0]: "False-negative rate"})
            fig.update_layout(height=340, margin=dict(t=10, b=10), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            if n_col:
                small_n = fair[fair[n_col[0]] < 100]
                if not small_n.empty:
                    st.warning(
                        f"⚠️ Categories with fewer than 100 true high-risk patients "
                        f"({', '.join(small_n[fair.columns[0]].astype(str))}) have unreliable FNR estimates "
                        f"and should not be over-interpreted as evidence of bias."
                    )
        st.dataframe(fair, use_container_width=True)
    else:
        missing_notice("fairness_audit_by_insurance.csv")

    st.divider()
    st.caption(
        "This dashboard reflects the deployed model, Gradient Boosting, validated via a patient-level "
        "train-test split (0 patient overlap), 5-fold cross-validation, hyperparameter tuning across all "
        "seven models compared, bootstrap significance testing, and this fairness audit."
    )
