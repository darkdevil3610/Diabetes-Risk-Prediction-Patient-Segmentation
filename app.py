"""
STREAMLIT APP — Diabetes Risk Predictor
Run: streamlit run app.py
Requires: model.pkl in the same directory
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings(
    "ignore",
    category=InconsistentVersionWarning
)

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>


  .stButton > button {
      background:#1D9E75;
      color:white;
      border-radius:10px;
      font-weight:600;
      width:100%;
      padding:0.6rem;
      border:none;
  }

  .stButton > button:hover {
      background:#0A7A57;
      color:white;
  }

  .result-high,
  .result-low,
  .result-med {
      padding:1rem;
      border-radius:8px;
      margin-top:1rem;
      color:#1F2937;
  }

  .result-high *,
  .result-low *,
  .result-med * {
      color:inherit !important;
  }

  .result-high {
      background:#FDEDEC;
      border-left:5px solid #C0392B;
  }

  .result-low {
      background:#EAFAF1;
      border-left:5px solid #1D9E75;
  }

  .result-med {
      background:#FEF9E7;
      border-left:5px solid #F39C12;
  }
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():

    model_path = Path(__file__).with_name("model.pkl")

    try:
        saved = joblib.load(model_path)

        scaler = None

        # If saved as dictionary
        if isinstance(saved, dict):
            model = saved.get("model")
            scaler = saved.get("scaler")

        # If only model saved
        else:
            model = saved

        return model, scaler, None

    except FileNotFoundError:
        return None, None, f"Missing file: {model_path}"

    except Exception as e:
        return None, None, str(e)


model, scaler, err = load_model()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("🩺 Diabetes Risk Predictor")

st.markdown("""
*Pima Indians Diabetes Database · Machine Learning Prediction System*
""")

st.markdown("---")

# ── ERROR HANDLING ────────────────────────────────────────────────────────────
if err:
    st.error(f"""
    ⚠️ Could not load the model file.

    Make sure `model.pkl` exists in the same folder as `app.py`

    Error:
    `{err}`
    """)
    st.stop()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.header("Patient Clinical Data")

st.sidebar.markdown("""
Enter the patient's medical values below:
""")

pregnancies = st.sidebar.slider(
    "Pregnancies",
    0, 17, 1
)

glucose = st.sidebar.slider(
    "Glucose (mg/dL)",
    44, 199, 100
)

bp = st.sidebar.slider(
    "Blood Pressure (mmHg)",
    24, 122, 70
)

skin = st.sidebar.slider(
    "Skin Thickness (mm)",
    7, 99, 20
)

insulin = st.sidebar.slider(
    "Insulin (μU/ml)",
    14, 846, 80
)

bmi = st.sidebar.number_input(
    "BMI",
    10.0, 70.0, 24.0,
    step=0.1
)

dpf = st.sidebar.number_input(
    "Diabetes Pedigree Function",
    0.05, 2.50, 0.20,
    step=0.01
)

age = st.sidebar.slider(
    "Age (years)",
    21, 81, 25
)

st.sidebar.markdown("---")

predict_btn = st.sidebar.button(
    "🔍 Predict Diabetes Risk",
    use_container_width=True
)

# ── PATIENT SUMMARY ───────────────────────────────────────────────────────────
st.subheader("📋 Patient Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Glucose", f"{glucose} mg/dL")
col2.metric("BMI", f"{bmi:.1f}")
col3.metric("Age", f"{age} yrs")
col4.metric("Pedigree", f"{dpf:.2f}")

st.markdown("---")

# ── PREDICTION ────────────────────────────────────────────────────────────────
if predict_btn:

    # Create input array
    input_arr = pd.DataFrame([{
    "Pregnancies": pregnancies,
    "Glucose": glucose,
    "BloodPressure": bp,
    "SkinThickness": skin,
    "Insulin": insulin,
    "BMI": bmi,
    "DiabetesPedigreeFunction": dpf,
    "Age": age
    }])

    # Apply scaler if available
    model_input = input_arr

    if scaler is not None:
        try:
            model_input = scaler.transform(input_arr)

        except Exception as e:
            st.warning(f"Scaler transform failed: {e}")

    # Prediction
    prediction = int(model.predict(model_input)[0])

    # Probability calculation
    probability = None

    if hasattr(model, "predict_proba"):

        proba = model.predict_proba(model_input)

        if proba.ndim == 2 and proba.shape[1] >= 2:
            probability = float(proba[0][1])

        else:
            probability = float(proba[0][0])

    elif hasattr(model, "decision_function"):

        score = float(model.decision_function(model_input)[0])

        probability = float(
            1 / (1 + np.exp(-score))
        )

    else:
        probability = float(prediction)

    # Safety clamp
    probability = max(0.0, min(1.0, probability))
    # ── RESULT DISPLAY ───────────────────────────────────────────────────────
    st.subheader("🎯 Prediction Result")

    model_name = model.__class__.__name__

    if probability >= 0.60:

        st.markdown(
    f"""
    <div class="result-high">
        <h3>🔴 HIGH RISK — {probability*100:.1f}% probability of Diabetes</h3>
        <p><strong>Model used:</strong> {model_name}</p>
        <p>⚠️ <strong>Recommendation:</strong><br>Immediate medical consultation advised.<br>Consider fasting glucose test, HbA1c screening, and lifestyle intervention.</p>
    </div>
    """,
    unsafe_allow_html=True
)
    elif probability >= 0.35:

       st.markdown(
    f"""
    <div class="result-med">
        <h3>🟡 MODERATE RISK — {probability*100:.1f}% probability of Diabetes</h3>
        <p><strong>Model used:</strong> {model_name}</p>
        <p>📋 <strong>Recommendation:</strong><br>Monitor blood glucose regularly.<br>Lifestyle changes like diet and exercise are strongly encouraged.</p>
    </div>
    """,
    unsafe_allow_html=True
)
    else:

        st.markdown(
    f"""
    <div class="result-low">
        <h3>🟢 LOW RISK — {probability*100:.1f}% probability of Diabetes</h3>
        <p><strong>Model used:</strong> {model_name}</p>
        <p>✅ <strong>Recommendation:</strong><br>Maintain a healthy lifestyle.<br>Annual check-ups recommended.</p>
    </div>
    """,
    unsafe_allow_html=True
)

    # ── PREDICTED CLASS ──────────────────────────────────────────────────────
    st.caption(
        f"Predicted class: `{prediction}` "
        "(1 = Diabetes, 0 = No Diabetes)"
    )

    # ── RISK FLAGS ───────────────────────────────────────────────────────────
    st.markdown("---")

    st.subheader("⚠️ Key Risk Flags")

    flags = []

    if glucose >= 140:
        flags.append(
            "🔴 High Glucose (≥140 mg/dL) — strongest predictor"
        )

    if bmi >= 30:
        flags.append(
            "🔴 Obese BMI — significant diabetes risk factor"
        )

    if age >= 45:
        flags.append(
            "🟡 Age ≥ 45 — risk increases substantially"
        )

    if dpf >= 0.6:
        flags.append(
            "🟡 High Pedigree Score — genetic risk present"
        )

    if insulin > 200:
        flags.append(
            "🟡 High Insulin — possible insulin resistance"
        )

    if not flags:
        flags.append(
            "🟢 No major clinical risk flags identified"
        )

    for flag in flags:
        st.write(flag)

    # ── DEBUG SECTION ────────────────────────────────────────────────────────
    with st.expander("🔍 Debug Info"):

        st.write("Input Data:", input_arr.to_dict(orient="records"))

        if 'proba' in locals():
            st.write("Raw Probabilities:", proba.tolist())

        st.write("Final Diabetes Probability:", probability)

# ── DEFAULT SCREEN ────────────────────────────────────────────────────────────
else:

    st.info("""
    👈 Enter patient data in the sidebar
    and click Predict to get a result.
    """)

    st.markdown("""
    ### How to use

    1. Enter patient clinical measurements
    2. Click **Predict Diabetes Risk**
    3. Review the prediction and recommendations

    ### Example Healthy Sample

    - Pregnancies: 0
    - Glucose: 85
    - Blood Pressure: 70
    - Skin Thickness: 20
    - Insulin: 79
    - BMI: 21
    - DPF: 0.2
    - Age: 24

    ⚕️ This tool is for educational and research purposes only.
    It should not replace professional medical diagnosis.
    """)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")

st.caption("""
Dataset: Pima Indians Diabetes Database
· UCI Machine Learning Repository
· 768 records
· 8 clinical features
""")
