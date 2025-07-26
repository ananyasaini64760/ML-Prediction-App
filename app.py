import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("model/trained_model.pkl", "rb"))

st.title("🧠 Machine Learning Prediction App")
st.write("Enter the details below to get a prediction:")

# User Inputs
age = st.slider("Age", 18, 70, 30)
salary = st.number_input("Salary (in ₹)", min_value=1000, max_value=1000000, value=50000)
gender = st.selectbox("Gender", ("Male", "Female"))
experience = st.slider("Years of Experience", 0, 40, 5)

# Convert gender to numeric
gender_encoded = 1 if gender == "Male" else 0

# Make prediction
if st.button("Predict"):
    features = np.array([[age, salary, gender_encoded, experience]])
    prediction = model.predict(features)[0]
    st.success(f"Predicted Output: {prediction}")
