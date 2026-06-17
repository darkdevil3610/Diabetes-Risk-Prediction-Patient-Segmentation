"""
STREAMLIT APP — Diabetes Risk Predictor
Run: streamlit run app.py
Requires: model.pkl and diabetes.csv in the same directory
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report, accuracy_score
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

# ── HIGH-END PREMIUM CLINICAL STYLING ──────────────────────────────────────────
st.markdown("""
<style>
    /* ── PRECISE SIDEBAR TRANSFORMATION ──────────────── */
    section[data-testid="stSidebar"] {
        background-color: #0F172A !important; /* Rich Dark Slate Background */
        border-right: 1px solid #1E293B;
    }

    /* Target labels and markdown text specifically, ignoring internal system icons */
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #E2E8F0 !important;
        font-family: 'Inter', system-ui, sans-serif;
    }

    section[data-testid="stSidebar"] label {
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        color: #94A3B8 !important;
    }

    /* Custom UI Slider Handles and Tracks */
    section[data-testid="stSidebar"] div[data-testid="stThumbValue"] {
        background-color: #0EA5E9 !important;
        color: white !important;
        border-radius: 6px !important;
        padding: 2px 6px !important;
        font-size: 0.8rem !important;
    }

    section[data-testid="stSidebar"] div[role="slider"] {
        background-color: #0EA5E9 !important;
        border: 2px solid #0EA5E9 !important;
    }

    /* Number Input Fields Inside Sidebar */
    section[data-testid="stSidebar"] input[type="number"] {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }

    section[data-testid="stSidebar"] button[data-testid="stNumberInputStepUp"],
    section[data-testid="stSidebar"] button[data-testid="stNumberInputStepDown"] {
        background-color: #1E293B !important;
        color: #94A3B8 !important;
        border: none !important;
    }

    /* ── MAIN CONTENT ELEMENTS ────────────────────────── */
    /* Premium gradient main action button */
    .stButton > button {
        background: linear-gradient(135deg, #0EA5E9 0%, #2563EB 100%);
        color: white !important;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
        padding: 0.7rem;
        border: none;
        transition: all 0.25s ease;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #0284C7 0%, #1D4ED8 100%);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.35);
        transform: translateY(-1px);
    }

    /* Theme-agnostic framing layout for main workspace metrics */
    div[data-testid="stMetric"] {
        background-color: rgba(15, 23, 42, 0.03); /* Soft semi-transparent tint */
        padding: 0.75rem 0.5rem !important; /* Slightly tighter padding */
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.2);
    }

    /* CRITICAL FIX: Prevent truncation (ellipses) and dynamically size text */
    div[data-testid="stMetricValue"] > div {
        color: inherit !important;
        font-size: calc(1.3rem + 0.3vw) !important; /* Responsive fluid size */
        white-space: normal !important; /* Allow wrapping if absolutely needed */
        text-overflow: clip !important; /* Force overflow removal */
        overflow: visible !important;
    }

    div[data-testid="stMetricLabel"] > div {
        color: rgba(100, 116, 139, 0.9) !important;
        font-weight: 500;
        font-size: 0.85rem !important; /* Slightly smaller label to maximize value space */
    }
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL & DATA ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = Path(__file__).with_name("model.pkl")
    try:
        saved = joblib.load(model_path)
        scaler = None
        if isinstance(saved, dict):
            model = saved.get("model")
            scaler = saved.get("scaler")
        else:
            model = saved
        return model, scaler, None
    except FileNotFoundError:
        return None, None, f"Missing file: {model_path}"
    except Exception as e:
        return None, None, str(e)

@st.cache_data
def load_data(file_buffer=None):
    if file_buffer is not None:
        return pd.read_csv(file_buffer)
    default_path = Path(__file__).with_name("diabetes.csv")
    if default_path.exists():
        return pd.read_csv(default_path)
    return None

model, scaler, err = load_model()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("🩺 Diabetes Risk Predictor")
st.markdown("*Pima Indians Diabetes Database · Machine Learning Prediction System*")
st.markdown("---")

# ── ERROR HANDLING ────────────────────────────────────────────────────────────
if err:
    st.error(f"⚠️ Could not load the model file.\n\nMake sure `model.pkl` exists in the same folder as `app.py` \n\nError: `{err}`")
    st.stop()

# ── NAVIGATION SIDEBAR ────────────────────────────────────────────────────────
app_mode = st.sidebar.radio("🧭 Navigation", ["🔍 Diagnostic Evaluator", "📊 Data Exploration", "📈 Model Performance"])
st.sidebar.markdown("---")

# ==============================================================================
# ── MODE 1: DIAGNOSTIC EVALUATOR ──────────────────────────────────────────────
# ==============================================================================
if app_mode == "🔍 Diagnostic Evaluator":

    # ── SIDEBAR (PREMIUM SLATE CONTROL PANEL) ──────────────────────────────────────
    st.sidebar.markdown("### 🧑‍⚕️ Patient Clinical Profile")
    st.sidebar.markdown("Adjust the verified metabolic markers below:")
    st.sidebar.markdown("---")

    pregnancies = st.sidebar.slider("Pregnancies", 0, 17, 1)
    glucose = st.sidebar.slider("Glucose (mg/dL)", 44, 199, 100)
    bp = st.sidebar.slider("Blood Pressure (mmHg)", 24, 122, 70)
    skin = st.sidebar.slider("Skin Thickness (mm)", 7, 99, 20)
    insulin = st.sidebar.slider("Insulin (μU/ml)", 14, 846, 80)
    bmi = st.sidebar.number_input("BMI", 10.0, 70.0, 24.0, step=0.1)
    dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.05, 2.50, 0.20, step=0.01)
    age = st.sidebar.slider("Age (years)", 21, 81, 25)

    st.sidebar.markdown("---")
    predict_btn = st.sidebar.button("🔍 Run Diagnostic Risk Evaluation", use_container_width=True)

    # ── PATIENT SUMMARY ───────────────────────────────────────────────────────────
    st.subheader("📋 Active Patient Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Glucose Level", f"{glucose} mg/dL")
    col2.metric("Body Mass Index", f"{bmi:.1f}")
    col3.metric("Patient Age", f"{age} yrs")
    col4.metric("Pedigree Value", f"{dpf:.2f}")

    st.markdown("---")

    # ── PREDICTION & SIMULATION ENGINE ────────────────────────────────────────────
    if "has_predicted" not in st.session_state:
        st.session_state.has_predicted = False
    if "saved_input" not in st.session_state:
        st.session_state.saved_input = None

    if predict_btn:
        st.session_state.has_predicted = True
        st.session_state.saved_input = pd.DataFrame([{
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": bp,
            "SkinThickness": skin,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age
        }])

    if st.session_state.has_predicted:
        input_arr = st.session_state.saved_input

        model_input = input_arr
        if scaler is not None:
            try:
                model_input = scaler.transform(input_arr)
            except Exception as e:
                st.warning(f"Scaler transform failed: {e}")

        prediction = int(model.predict(model_input)[0])
        probability = None

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(model_input)
            probability = float(proba[0][1]) if (proba.ndim == 2 and proba.shape[1] >= 2) else float(proba[0][0])
        elif hasattr(model, "decision_function"):
            score = float(model.decision_function(model_input)[0])
            probability = float(1 / (1 + np.exp(-score)))
        else:
            probability = float(prediction)

        probability = max(0.0, min(1.0, probability))

        # ── POPUP TOAST ALERTS ────────────────────────────────────────────────────
        if predict_btn:
            if probability >= 0.60:
                st.toast(f"🔴 CRITICAL ATTENTION REQUIRED: Patient evaluated at {probability*100:.1f}% risk tier.", icon="⚠️")
            elif probability >= 0.35:
                st.toast(f"🟡 ELEVATED ALERT: Patient falling into Moderate Risk limits ({probability*100:.1f}%).", icon="📋")
            else:
                st.toast(f"🟢 STABLE METRICS: Safe risk baseline computed profile ({probability*100:.1f}%).", icon="✅")

        # ── RESULT DISPLAY ───────────────────────────────────────────────────────
        st.subheader("🎯 Diagnostic Profile Results")
        model_name = model.__class__.__name__

        if probability >= 0.60:
            with st.container(border=True):
                st.error(f"### 🔴 CRITICAL RANGE — {probability*100:.1f}% Risk Probability")
                st.markdown(f"**Analytics Engine:** `{model_name}`")
                st.markdown("⚠️ **Clinical Directive:** Immediate physician review recommended. Consider fast-tracking HbA1c screening assessments alongside continuous metabolic checks.")
        elif probability >= 0.35:
            with st.container(border=True):
                st.warning(f"### 🟡 BORDERLINE ELEVATED — {probability*100:.1f}% Risk Probability")
                st.markdown(f"**Analytics Engine:** `{model_name}`")
                st.markdown("📋 **Clinical Directive:** Monitor blood glucose benchmarks routinely. Structured lifestyle adjustments, focused activity metrics, and dietary changes are indicated.")
        else:
            with st.container(border=True):
                st.success(f"### 🟢 LOW RISK METRICS — {probability*100:.1f}% Risk Probability")
                st.markdown(f"**Analytics Engine:** `{model_name}`")
                st.markdown("✅ **Clinical Directive:** Maintain baseline optimal physiological metrics. Standard wellness evaluations and routine check-ups are sufficient.")

        st.caption(f"Evaluated Target Class: `{prediction}` (1 = Diabetes Risk Flagged, 0 = Within Normal Baseline)")

        # ── LIGHTWEIGHT INTERACTIVE SEGMENTATION SLIDER ───────────────────────────
        st.markdown("---")
        st.subheader("📊 Therapeutic Optimization Sandbox")
        st.markdown("Isolate and evaluate how adjusting this specific patient's glucose target parameter scales down the risk index in real-time:")

        base_glucose = int(input_arr["Glucose"].iloc[0])

        target_glucose = st.slider("Simulate Lower Glucose Target (mg/dL)", 70, int(max(base_glucose, 100)), base_glucose)

        sim_arr = input_arr.copy()
        sim_arr["Glucose"] = target_glucose
        sim_input = scaler.transform(sim_arr) if scaler is not None else sim_arr

        if hasattr(model, "predict_proba"):
            sim_proba = model.predict_proba(sim_input)
            sim_probability = float(sim_proba[0][1]) if (sim_proba.ndim == 2 and sim_proba.shape[1] >= 2) else float(sim_proba[0][0])
        else:
            sim_probability = float(model.predict(sim_input)[0])

        sim_probability = max(0.0, min(1.0, sim_probability))

        s_col1, s_col2 = st.columns(2)
        s_col1.metric("Original Calculated Risk", f"{probability*100:.1f}%")
        s_col2.metric("Simulated Mitigated Risk", f"{sim_probability*100:.1f}%",
                      delta=f"-{(probability - sim_probability)*100:.1f}%" if probability > sim_probability else None)

        # ── RISK FLAGS ───────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("⚠️ Core Structural Comorbidities")

        flags = []
        if base_glucose >= 140: flags.append("🔴 **High Glucose Threshold Exception (≥140 mg/dL)** — Prominent statistical model predictor variable.")
        if float(input_arr["BMI"].iloc[0]) >= 30: flags.append("🔴 **Clinical Obesity Metric (BMI ≥ 30)** — Strongly correlated with elevated metabolic insulin resistance values.")
        if int(input_arr["Age"].iloc[0]) >= 45: flags.append("🟡 **Age Baseline Vector (≥45)** — Statistical acceleration threshold milestone.")
        if float(input_arr["DiabetesPedigreeFunction"].iloc[0]) >= 0.6: flags.append("🟡 **Elevated Pedigree Function Index** — Notable inherited genetic risk marker trends.")
        if int(input_arr["Insulin"].iloc[0]) > 200: flags.append("🟡 **Hyperinsulinemia Metric (>200 μU/ml)** — Potential sign of prolonged beta-cell strain.")
        if not flags: flags.append("🟢 **Optimal Functional Baselines** — No major standard risk parameters triggered.")

        for flag in flags:
            st.markdown(flag)

        # ── DEBUG SECTION ────────────────────────────────────────────────────────
        with st.expander("🔍 System Telemetry Log"):
            st.write("Input Vector Array:", input_arr.to_dict(orient="records"))
            st.write("Calculated Raw Floating-Point Probability Value:", probability)

    # ── DEFAULT SCREEN ────────────────────────────────────────────────────────────
    else:
        st.info("👈 Please input the required medical markers in the Patient Clinical Profile sidebar and hit evaluate.")
        st.markdown("""
        ### How to Interact with the System
        1. Fill in the clinical data sliders located inside the dark control workspace panel to the left.
        2. Select **Run Diagnostic Risk Evaluation** to parse the inputs.
        3. Study the downstream stratified metrics output and real-time optimization sandbox options.

        ### Standard Dataset Healthy Sample Target Values
        * **Pregnancies:** 0  |  **Glucose Baseline:** 85 mg/dL  |  **Blood Pressure:** 70 mmHg
        * **Skin Thickness Value:** 20 mm  |  **Serum Insulin:** 79 μU/ml  |  **BMI Index:** 21.2
        * **Pedigree Score:** 0.24  |  **Age:** 24 years old

        ⚕️ *Disclaimer: This layout architecture is provided strictly for educational modeling validation tasks. It must never substitute direct human practitioner medical interventions.*
        """)

# ==============================================================================
# ── MODE 2: DATA EXPLORATION ──────────────────────────────────────────────────
# ==============================================================================
elif app_mode == "📊 Data Exploration":
    st.subheader("📊 Dataset Exploration & Visualizations")
    st.markdown("Explore the underlying **Pima Indians Diabetes Database** used to train the machine learning system.")

    uploaded_file = st.sidebar.file_uploader("Upload `diabetes.csv` to view EDA (optional if available locally)", type=["csv"])
    df = load_data(uploaded_file)

    if df is not None:
        st.markdown("### 1. Dataset Overview & Statistics")
        st.write("First 5 rows of the dataset:")
        st.dataframe(df.head())

        st.write("Dataset Summary Statistics:")
        st.dataframe(df.describe().T)
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Feature Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(df.corr(), annot=False, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

        with col2:
            if "Outcome" in df.columns:
                st.markdown("#### Target Distribution")
                fig2, ax2 = plt.subplots(figsize=(6, 5))
                sns.countplot(x='Outcome', data=df, palette={'0': '#5DCAA5', '1': '#F0997B'}, ax=ax2)
                st.pyplot(fig2)

        st.markdown("---")
        st.markdown("### 2. Feature Distributions")
        features = [c for c in df.columns if c != 'Outcome']
        fig3, axes = plt.subplots(int(np.ceil(len(features)/2)), 2, figsize=(12, 14))
        axes = axes.flatten()
        for i, col in enumerate(features):
            if "Outcome" in df.columns:
                sns.histplot(data=df, x=col, hue="Outcome", kde=True, ax=axes[i], palette="husl", element="step")
            else:
                sns.histplot(df[col], kde=True, ax=axes[i], color='#0EA5E9')
            axes[i].set_title(f'Distribution of {col}', fontweight='bold')
        for j in range(i + 1, len(axes)):
            fig3.delaxes(axes[j])
        plt.tight_layout()
        st.pyplot(fig3)

        st.markdown("---")
        st.markdown("### 3. Outlier Analysis (Box Plots)")
        fig4, ax4 = plt.subplots(figsize=(12, 7))
        sns.boxplot(data=df[features], ax=ax4, palette="Set2")
        ax4.set_title("Box plot of Features (Checking for Outliers)", fontweight='bold')
        plt.xticks(rotation=45)
        st.pyplot(fig4)
    else:
        st.info("⚠️ `diabetes.csv` not found in the local directory. Please upload the dataset via the sidebar to view the interactive visualizations.")


# ==============================================================================
# ── MODE 3: MODEL PERFORMANCE ─────────────────────────────────────────────────
# ==============================================================================
elif app_mode == "📈 Model Performance":
    st.subheader("📈 Machine Learning Model Performance")
    st.markdown("Evaluate the active classification model against the dataset and compare it to standard baselines.")

    uploaded_file = st.sidebar.file_uploader("Upload `diabetes.csv` for evaluation", type=["csv"])
    df = load_data(uploaded_file)

    if df is not None and "Outcome" in df.columns:
        X = df.drop("Outcome", axis=1)
        y = df["Outcome"]

        # Scale if scaler is present
        X_scaled = scaler.transform(X) if scaler else X

        # Predictions & Current Accuracy
        y_pred = model.predict(X_scaled)
        current_acc = accuracy_score(y, y_pred)

        # --- MODEL COMPARISON TABLE ---
        st.markdown("#### 🏆 Model Accuracies")

        # Exact data provided for the baselines
        comparison_data = {
            "Rank": [1, 2, 3, 4],
            "Model Architecture": [
                "Random Forest",
                "SVM (RBF Kernel)",
                "Decision Tree",
                "Logistic Regression"
            ],
            "Cross-Validation Accuracy (%)": [
                "97.22",
                "94.44",
                "93.52",
                "87.96"
            ]
        }
        comp_df = pd.DataFrame(comparison_data)

        # Display the dataframe
        st.dataframe(comp_df, use_container_width=True, hide_index=True)


        st.markdown("---")

        # --- VISUAL METRICS ---
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 1. Confusion Matrix of Best Model (Random Forest)")
            cm = confusion_matrix(y, y_pred)
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, cbar=False)
            ax_cm.set_xlabel('Predicted')
            ax_cm.set_ylabel('Actual')
            st.pyplot(fig_cm)

        with col2:
            st.markdown("#### 2. Classification Report")
            report = classification_report(y, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.2f}"))

        st.markdown("---")

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("#### 3. ROC Curve")
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_scaled)[:, 1]
                fpr, tpr, _ = roc_curve(y, y_prob)
                roc_auc = auc(fpr, tpr)

                fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
                ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax_roc.set_xlabel('False Positive Rate')
                ax_roc.set_ylabel('True Positive Rate')
                ax_roc.legend(loc="lower right")
                st.pyplot(fig_roc)
            else:
                st.info("ROC Curve unavailable: Model does not support probability predictions.")

        with col4:
            st.markdown("#### 4. Feature Importance")
            # Try Tree-based models
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                imp_df = pd.DataFrame({"Feature": X.columns, "Importance": importances}).sort_values(by="Importance", ascending=False)
                fig_imp, ax_imp = plt.subplots(figsize=(5, 4))
                sns.barplot(x="Importance", y="Feature", data=imp_df, ax=ax_imp, palette="viridis")
                st.pyplot(fig_imp)
            # Try Linear models
            elif hasattr(model, "coef_"):
                importances = model.coef_[0]
                imp_df = pd.DataFrame({"Feature": X.columns, "Coefficient": importances}).sort_values(by="Coefficient", ascending=False)
                fig_imp, ax_imp = plt.subplots(figsize=(5, 4))
                sns.barplot(x="Coefficient", y="Feature", data=imp_df, ax=ax_imp, palette="coolwarm")
                st.pyplot(fig_imp)
            else:
                st.info("Feature importance unavailable for this model type.")

    else:
        st.info("⚠️ Target column 'Outcome' not found or dataset not loaded. Upload `diabetes.csv` to compute metrics.")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Core Infrastructure Metadata: Pima Indians Diabetes Database · Managed through UCI Machine Learning Repository Hub · 768 patient indices with 8 continuous features.")
