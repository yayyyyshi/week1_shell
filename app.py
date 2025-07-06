import streamlit as st
import joblib
import numpy as np
import pandas as pd
from utils.preprocessor import preprocess_input

# ---- Page Setup ----
st.set_page_config(
    page_title="GHG Emissions Predictor 🌍",
    page_icon="🌱",
    layout="centered"
)

# ---- Load model and preprocessor ----
model = joblib.load('models/LR_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# ---- Sidebar Branding ----
with st.sidebar:
    st.image("images/Fire relief - Ecosystems not emissions poster.png", width=120)
    st.markdown("### Developed by **Yashika Verma**")
    st.markdown("🧪 AICTE Internship — GHG Emissions Modeling")
    st.markdown("🔗 [GitHub](https://github.com/yayyyyshi/week1_shell)")
    st.markdown("💻 Powered by Streamlit")

# ---- App Title and Subtitle ----
st.title("🌿 GHG Supply Chain Emissions Prediction")
st.subheader("💡 Predict emissions based on data quality metrics and material inputs")

st.markdown("""
Use the form below to input relevant parameters and predict the **Supply Chain Emission Factor with Margins**.
""")

# ---- Input Form ----
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        substance = st.selectbox("🌫️ Substance", ['carbon dioxide', 'methane', 'nitrous oxide', 'other GHGs'])
        unit = st.selectbox("📐 Unit", [
            'kg/2018 USD, purchaser price',
            'kg CO2e/2018 USD, purchaser price'
        ])
        source = st.selectbox("🏭 Source", ['Commodity', 'Industry'])

    with col2:
        supply_wo_margin = st.number_input("🧮 Emission Factor (Without Margin)", min_value=0.0)
        margin = st.number_input("➕ Margin on Emissions", min_value=0.0)

    st.markdown("### 📊 Data Quality Scores")
    dq_reliability = st.slider("🧷 Reliability", 0.0, 1.0, 0.5)
    dq_temporal = st.slider("⏳ Temporal Correlation", 0.0, 1.0, 0.5)
    dq_geo = st.slider("🌍 Geographical Correlation", 0.0, 1.0, 0.5)
    dq_tech = st.slider("⚙️ Technological Correlation", 0.0, 1.0, 0.5)
    dq_data = st.slider("📥 Data Collection", 0.0, 1.0, 0.5)

    submit = st.form_submit_button("🎯 Predict Emissions")

# ---- Prediction ----
if submit:
    input_data = {
        'Substance': substance,
        'Unit': unit,
        'Supply Chain Emission Factors without Margins': supply_wo_margin,
        'Margins of Supply Chain Emission Factors': margin,
        'DQ ReliabilityScore of Factors without Margins': dq_reliability,
        'DQ TemporalCorrelation of Factors without Margins': dq_temporal,
        'DQ GeographicalCorrelation of Factors without Margins': dq_geo,
        'DQ TechnologicalCorrelation of Factors without Margins': dq_tech,
        'DQ DataCollection of Factors without Margins': dq_data,
        'Source': source,
    }

    input_df = preprocess_input(pd.DataFrame([input_data]))
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    st.markdown("---")
    st.markdown("## ✅ Prediction Result")
    st.metric(label="Predicted Emission Factor (with Margin)", value=f"{prediction[0]:.4f} kg CO2e")
    st.success("Prediction completed successfully!")

# ---- Footer ----
st.markdown("""---  
<p style='text-align: center; font-size: 0.9em; color: gray'>
Made with ❤️ by Yashika Verma | Shell Internship Project 2025 | Streamlit Powered
</p>""", unsafe_allow_html=True)
