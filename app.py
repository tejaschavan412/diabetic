import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Diabetes Risk Assessment", 
    layout="wide", 
    page_icon="🩺"
)

# --- 2. Sleek Custom CSS ---
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        color: #1F618D;
    }
    div.stButton > button:first-child {
        background-color: #2874A6;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 18px;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease 0s;
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        background-color: #1B4F72;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# --- 3. Load Machine Learning Artifacts ---
@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("sc.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

try:
    model, scaler = load_artifacts()
    models_loaded = True
except FileNotFoundError:
    models_loaded = False
    st.error("⚠️ System Error: 'model.pkl' or 'sc.pkl' not found in the current directory.")

# --- 4. Gauge Chart Helper Function ---
def create_gauge_chart(probability):
    """Creates a color-coded Plotly gauge chart based on probability."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        number = {'suffix': "%", 'font': {'size': 40, 'color': '#1F618D'}},
        title = {'text': "Risk Probability", 'font': {'size': 24, 'color': '#1F618D'}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "rgba(0,0,0,0)"},  # Hide the default bar to use the threshold line
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': "#ABEBC6"},     # Safe (Green)
                {'range': [30, 50], 'color': "#F9E79F"},    # Elevated (Yellow)
                {'range': [50, 75], 'color': "#F5CBA7"},    # Moderate Risk (Orange)
                {'range': [75, 100], 'color': "#F1948A"}    # High Risk (Red)
            ],
            'threshold': {
                'line': {'color': "#1F618D", 'width': 6},
                'thickness': 0.8,
                'value': probability * 100
            }
        }
    ))
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# --- 5. Dashboard Header ---
st.title("🩺 Predictive Health Dashboard")
st.markdown("Enter patient diagnostics below to assess the statistical probability of diabetes.")
st.divider()

# --- 6. Layout Setup ---
st.sidebar.header("👤 Patient Demographics")
age = st.sidebar.number_input("Age (years)", min_value=21, max_value=120, value=30, step=1)
bmi = st.sidebar.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
pregnancies = st.sidebar.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1, step=1)

st.header("🔬 Clinical Metrics")
col1, col2 = st.columns(2)

with col1:
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=0.0, max_value=300.0, value=120.0)
    insulin = st.number_input("Insulin Level (IU/mL)", min_value=0.0, max_value=900.0, value=80.0)

with col2:
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, max_value=200.0, value=72.0)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=20.0)

dpf = st.slider("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)

st.markdown("<br>", unsafe_allow_html=True)

# --- 7. Prediction Execution ---
if st.button("Run Diagnostic Assessment"):
    if models_loaded:
        feature_columns = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        
        input_data = pd.DataFrame([[
            pregnancies, glucose, blood_pressure, skin_thickness, 
            insulin, bmi, dpf, age
        ]], columns=feature_columns)
        
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0]
        risk_prob = probability[1]
        
        st.divider()
        st.header("📋 Assessment Results")
        
        col_res1, col_res2 = st.columns([1, 1])
        
        with col_res1:
            st.markdown("<br><br>", unsafe_allow_html=True)
            if prediction == 1:
                st.error("### 🚨 High Risk Detected")
                st.write("The model indicates a high statistical likelihood of diabetes based on the provided clinical metrics. Immediate clinical follow-up and diagnostic confirmation are highly recommended.")
            else:
                st.success("### ✅ Low Risk Detected")
                st.write("The model indicates a low statistical likelihood of diabetes. The patient's metrics fall within generally acceptable ranges based on the training data.")
            
        with col_res2:
            # Display the interactive Plotly Gauge
            st.plotly_chart(create_gauge_chart(risk_prob), use_container_width=True)
            
    else:
        st.warning("Prediction engine offline. Missing required model artifacts.")