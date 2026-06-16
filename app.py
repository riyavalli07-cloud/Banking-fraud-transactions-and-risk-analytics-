import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("fraud_model.pkl","rb"))

st.title("🏦 AI Banking Fraud Detection")

amount = st.number_input("Transaction Amount")

payment = st.selectbox(
    "Payment Channel",
    ["UPI","Card","Net Banking"]
)

auth = st.selectbox(
    "Authentication Type",
    ["OTP","Password","Biometric"]
)

if st.button("Predict"):

    data = pd.DataFrame({
        "amount":[amount],
        "payment_channel":[payment],
        "authentication_type":[auth]
    })

    result = model.predict(data)

    if result[0] == 1:
        st.error("Fraud Transaction Detected")
    else:
        st.success("Normal Transaction")