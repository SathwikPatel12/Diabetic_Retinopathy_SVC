
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from streamlit_lottie import st_lottie

# -------------------------
# Load Model (with caching)
# -------------------------
import cloudpickle

@st.cache_resource
def load_model():
    with open("retinopathy_model_svm.pkl", "rb") as f:
        return cloudpickle.load(f)
model = load_model()

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Diabetic Retinopathy Detection", layout="centered")

# -------------------------
# Custom CSS Styling (Improvement 1)
# -------------------------
st.markdown("""
    <style>
        .main {
            background-color: #fefefe;
        }
        h1 {
            font-family: 'Helvetica Neue', sans-serif;
            color: #2c3e50;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 10px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------
# App Title & Description
# -------------------------
st.title("👁️🩺 Diabetic Retinopathy Prediction App")
st.markdown("This app predicts whether a person shows signs of diabetic retinopathy based on input health features.")


# -------------------------
# Lottie Animation Display (Modified)
# ✅ Show first 2 animations side by side, no dropdown
# ✅ Purpose:
# Adds visual interest and professional medical feel.
# Makes your app more welcoming and modern.
# -------------------------



from streamlit_lottie import st_lottie
import requests

def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Define URLs for first two animations
urls = [
    "https://assets1.lottiefiles.com/packages/lf20_3vbOcw.json",  # Hello Bot
    "https://assets2.lottiefiles.com/packages/lf20_tutvdkg0.json"  # Doctor
]

# Updated heading (replaces "🎞️ Medical Animations")
st.markdown("### 👋 Welcome to Your Eye Health Companion")

# Load both animations
animations = [load_lottie_url(url) for url in urls]

# Display them side by side with slightly tighter spacing
col1, col2 = st.columns([1, 1], gap="small")

with col1:
    if animations[0]:
        st_lottie(animations[0], height=250)
    else:
        st.warning("⚠️ Animation couldn't load.")

with col2:
    if animations[1]:
        st_lottie(animations[1], height=250)
    else:
        st.warning("⚠️ Animation couldn't load.")



# -------------------------
# Sidebar Layout and Info
# -------------------------
with st.sidebar:
    st.title("🧭 Navigation")

    with st.expander("ℹ️ About This App"):
        st.markdown("""
        This tool predicts **Diabetic Retinopathy (DR)** using vitals and simple inputs.

        - ✅ **Purpose**: Predict Diabetic Retinopathy presence
        - ⚙️ **Model**: SVC, Support Vector Machine Classifier (scikit-learn)
        - 📊 Features: Age, BP, Cholesterol  
        - 📐 **Derived Features**: Pulse Pressure & MAP
        - 🧠 **Built with**: Streamlit + Joblib
        """)

    st.markdown("---")

    st.markdown("### 🧪 Input Guide")
    st.markdown("""
    - **Age**: 30–105 years  
    - **Systolic BP**: 70–130 mmHg  
    - **Diastolic BP**: 60–120 mmHg  
    - **Cholesterol**: 70–130 mg/dL
    """)

    st.markdown("---")
    st.markdown("📁 [GitHub Source](https://github.com/SathwikPatel12/Diabetic_Retinopathy_apps)")
   


# -------------------------
# Input Form
# Form-based input layout
# -------------------------
with st.form("input_form"):
    st.subheader("📝 Enter Patient Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input('Age', min_value=30, max_value=105, value=50, help="Enter the patient's age in years (30–105)")
        systolic_bp = st.number_input('Systolic Blood Pressure', min_value=70.0, max_value=130.0, value=120.0,
                                      help="The top number of blood pressure (normal ~120 mmHg)")

    with col2:
        cholesterol = st.number_input('Cholesterol Level', min_value=70.0, max_value=130.0, value=90.0, help="Measured in mg/dL; higher levels may increase risk")
        diastolic_bp = st.number_input('Diastolic Blood Pressure', min_value=60.0, max_value=120.0, value=80.0,
                                      help="The bottom number of blood pressure (normal ~80 mmHg)")

    submitted = st.form_submit_button("🔍 Predict")


# -------------------------
# Prediction
# -------------------------
if submitted:

    # Prepare input DataFrame, Original features only
    input_df = pd.DataFrame([{
      'age': age,
      'systolic_bp': systolic_bp,
      'diastolic_bp': diastolic_bp,
      'cholesterol': cholesterol
    }])

    # Make Prediction using pipeline (which handles feature engineering internally)
    prediction = model.predict(input_df)[0]
    pred_proba = model.predict_proba(input_df)[0]
    confidence = pred_proba[prediction]

    # -------------------------
    # Derived Features Box (just for user display — not used by model)
    # -------------------------
    pulse_pressure = systolic_bp - diastolic_bp
    mean_arterial_pressure = (systolic_bp + 2 * diastolic_bp) / 3

    # Derived Features Box
    # Display derived features (optional)
    # Detect Streamlit theme (light or dark)
    theme = st.get_option("theme.base")
    is_dark = theme == "dark"

    # Set background color accordingly
    bg_color = "#2b2b2b" if is_dark else "#f0f0f5"
    text_color = "#ffffff" if is_dark else "#000000"

    # Render the styled box
    st.markdown(f"""
        <div style='
            padding: 10px;
            background-color: {bg_color};
            color: {text_color};
            border-radius: 10px;
            margin-top: 10px;
        '>
            <b>Pulse Pressure:</b> {pulse_pressure:.2f} mmHg<br>
            <b>Mean Arterial Pressure:</b> {mean_arterial_pressure:.2f} mmHg
        </div>
    """, unsafe_allow_html=True)


    # -------------------------
    # Display prediction result    (Improvement 2)
    # -------------------------
    st.markdown("### 🔍 Prediction Result")
    #if prediction == 1:
    #    st.error(f"🧪 The model predicts **presence** of Diabetic Retinopathy (Confidence: {confidence:.2f})")
    #else:
    #    st.success(f"✅ The model predicts **no signs** of Diabetic Retinopathy (Confidence: {confidence:.2f})")
    
    if prediction == 1:
        st.error(f"🧪 The model predicts **presence** of Diabetic Retinopathy (Confidence: {confidence:.2f})")

        with st.expander("📢 What to Do Next?"):
            st.markdown("""
            - 👨‍⚕️ **Please consult an eye care specialist or diabetologist immediately.**
            - 🏥 Early treatment can help prevent vision loss.
            - 📘 You can read more at:
                - [American Diabetes Association](https://diabetes.org/health-wellness/eye-health)
                - [WHO on Diabetic Retinopathy](https://www.who.int/news-room/fact-sheets/detail/blindness-and-visual-impairment)
                - [Find a Retina Specialist Near You](https://www.centreforsight.net/eye-specialists-near-me)
            - 💊 Discuss your medications, blood sugar control, and eye care plan with a certified provider.
            """)

    else:
        st.success(f"✅ The model predicts **no signs** of Diabetic Retinopathy (Confidence: {confidence:.2f})")
    
        with st.expander("💡 Wellness Tip"):
            st.markdown("""
            - 👁️ It's still important to get your eyes checked **annually**.
            - 🥗 Maintain a healthy diet and regular exercise.
            - 📖 Learn about prevention: [NIH Diabetic Eye Disease Info](https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/diabetic-retinopathy)
            """)
 
    
    # Confidence Progress, Add a Progress Bar for Confidence
    #st.write("📊 Model Confidence:")
    #st.progress(confidence)
    # -------------------------
    # Confidence Gauge Meter (Improvement 2, advanced to just above progress bar)
    # -------------------------
    st.markdown("### 📊 Model Confidence Level")

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        #title={'text': "DR Risk (%)"},
        title={'text': "Confidence in DR Presence (%)" if prediction == 1 else "Confidence in No DR (%)"},

        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "green" if prediction == 0 else "red"},
            'steps': [
                {'range': [0, 50], 'color': "#d4edda"},
                {'range': [50, 75], 'color': "#fff3cd"},
                {'range': [75, 100], 'color': "#f8d7da"},
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': confidence * 100
            }
        }
    ))

    st.plotly_chart(gauge)

    
 
    # Download Prediction Report
    report = f"""
Prediction: {"DR Present" if prediction else "No DR"}
Confidence: {confidence:.2f}
Pulse Pressure: {pulse_pressure:.2f}
Mean Arterial Pressure: {mean_arterial_pressure:.2f}
"""
    st.download_button("📄 Download Report", report, file_name="dr_prediction_report.txt")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption("👨‍⚕️ Created with ❤️ by Sathwik Patel using Streamlit")
