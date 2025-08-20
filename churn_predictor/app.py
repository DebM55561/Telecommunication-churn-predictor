# app.py
import streamlit as st
import pandas as pd
import joblib

# Load trained model (after training, save it with joblib)
model = joblib.load("model.pkl")

st.title("Telecom Customer Churn Prediction")

# Collect inputs from user
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", min_value=0)
monthly = st.number_input("Monthly Charges", min_value=0.0)
total = st.number_input("Total Charges", min_value=0.0)
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
phone = st.selectbox("Phone Service", ["Yes", "No"])
multilines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
online_sec = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_bak = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_prot = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_sup = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
stream_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
stream_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

# Make prediction
if st.button("Predict Churn"):
    input_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "InternetService": internet,
        "Contract": contract,
        "PaymentMethod": payment,
        "PaperlessBilling": paperless,
        "PhoneService": phone,
        "MultipleLines": multilines,
        "OnlineSecurity": online_sec,
        "OnlineBackup": online_bak,
        "DeviceProtection": device_prot,
        "TechSupport": tech_sup,
        "StreamingTV": stream_tv,
        "StreamingMovies": stream_movies
    }])

    proba = model.predict_proba(input_data)[0][1]
    pred = model.predict(input_data)[0]

    st.write("### Result:")
    if pred == 1:
        st.error(f"⚠️ Customer likely to churn (probability {proba:.2f})")
    else:
        st.success(f"✅ Customer likely to stay (probability {proba:.2f})")
