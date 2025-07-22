import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import io

# Load model and encoders using joblib
joblib.dump(trained_model, "best_model.pkl")
joblib.dump(trained_columns, "trained_columns.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")


model = load_pickle("best_model.pkl")
trained_columns = load_pickle("trained_columns.pkl")
label_encoders = load_pickle("label_encoders.pkl")

# USD to INR conversion rate
USD_TO_INR = 83.5

# Page config
st.set_page_config(page_title="💼 Salary Predictor", layout="centered")
st.title("💼 Employee Salary Predictor")
st.markdown("Estimate your salary based on your profile.")

# --- Custom CSS styling ---
custom_css = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    padding: 2rem;
    color: white;
}
.block-container {
    backdrop-filter: blur(10px);
    background-color: rgba(0, 0, 0, 0.3);
    border-radius: 15px;
    padding: 2rem;
}
h1, h2, h3 {
    color: #ffffff;
    text-shadow: 1px 1px 2px #000;
}
.stButton > button, .stDownloadButton > button {
    background-color: #2c3e50;
    color: white;
    font-weight: bold;
    border-radius: 8px;
    padding: 10px 20px;
    border: none;
    box-shadow: 2px 2px 4px #00000055;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- Tabs ---
tabs = st.tabs(["🏠 Home", "📊 Predict Salary", "📤 Upload & Predict"])

# --- Home Tab ---
with tabs[0]:
    st.title("🏠 Welcome to Salary Prediction App")
    st.markdown("""
        This app helps you predict:
        - 💰 Annual and Monthly Salary
        - 📊 For Individuals or Bulk CSV Upload

        Powered by a trained **Random Forest** model (Accuracy: 0.84) using features like:
        - 👤 Age
        - 🎓 Education Level
        - 💼 Job Title
        - 🚻 Gender
        - ⏱ Hours Worked Per Week
    """)

# --- Individual Prediction Tab ---
with tabs[1]:
    st.title("📊 Predict Your Salary")

    age = st.sidebar.slider("Age", 18, 75, 30)
    education = st.selectbox("Select your education level", label_encoders["education"].classes_)
    job_title = st.selectbox("Select your job title", label_encoders["job_title"].classes_)
    gender = st.selectbox("Select your gender", label_encoders["gender"].classes_)
    experience = st.slider("Years of experience", min_value=0, max_value=50, value=2)
    hours = st.slider("Hours worked per week", min_value=1, max_value=100, value=40)

    if st.button("🔍 Predict Salary"):
        try:
            input_data = pd.DataFrame([{
                "age": age,
                "education": label_encoders["education"].transform([education])[0],
                "job_title": label_encoders["job_title"].transform([job_title])[0],
                "gender": label_encoders["gender"].transform([gender])[0],
                "experience": experience,
                "hours_per_week": hours
            }])
            input_data = input_data.reindex(columns=trained_columns)

            annual_usd = model.predict(input_data)[0]
            annual_usd = np.clip(annual_usd, 18000, 90000)
            annual_inr = annual_usd * USD_TO_INR
            monthly_inr = annual_inr / 12

            st.success(f"💰 Annual Salary: ₹{annual_inr:,.2f} (≈ ${annual_usd:,.2f})")
            st.info(f"📆 Monthly Salary: ₹{monthly_inr:,.2f}")

            st.warning("📢 Disclaimer: This is only an estimate and may not reflect actual salary values.")

            fig = px.line(
                x=list(range(0, 21)),
                y=[monthly_inr * (1 + 0.05 * i) for i in range(21)],
                labels={"x": "Years of Experience", "y": "Predicted Monthly Salary (INR)"},
                title="📈 Salary Growth Over Time"
            )
            st.plotly_chart(fig)

            report = io.StringIO()
            report.write("Salary Prediction Report\n")
            report.write("-----------------------------\n")
            report.write(f"Age: {age}\n")
            report.write(f"Education: {education}\n")
            report.write(f"Job Title: {job_title}\n")
            report.write(f"Gender: {gender}\n")
            report.write(f"Experience: {experience}\n")
            report.write(f"Hours/Week: {hours}\n")
            report.write(f"Annual Salary (INR): ₹{annual_inr:,.2f}\n")
            report.write(f"Monthly Salary (INR): ₹{monthly_inr:,.2f}\n")

            st.download_button("📥 Download Salary Report", report.getvalue(), file_name="salary_report.txt")

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")

# --- Upload Prediction Tab ---
with tabs[2]:
    st.title("📤 Upload CSV to Predict Salaries")
    st.markdown("""
    **Expected CSV Columns**: age, education, job_title, gender, experience, hours_per_week
    """)

    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        try:
            df = pd.read_csv(file)

            for col in ["education", "job_title", "gender"]:
                if col in df.columns:
                    df[col] = label_encoders[col].transform(df[col])

            df = df.reindex(columns=trained_columns)
            df["Annual_USD"] = model.predict(df)
            df["Annual_INR"] = df["Annual_USD"] * USD_TO_INR
            df["Monthly_INR"] = df["Annual_INR"] / 12

            st.success("✅ Predictions completed.")
            st.dataframe(df)

            download = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV", download, file_name="salary_predictions.csv")

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

# --- Footer ---
st.markdown("""
<br><hr>
<div style='text-align: center; color: #ccc;'>
© 2025 Salary Prediction App | Built by Sanjiv | Powered by Streamlit and scikit-learn
</div>
""", unsafe_allow_html=True)
