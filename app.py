import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Simulate IoT Sensor Data
def generate_sensor_data(n=100):
    np.random.seed(42)
    data = {
        "Temperature": np.random.uniform(20, 100, n),
        "Vibration": np.random.uniform(0.1, 5.0, n),
        "Pressure": np.random.uniform(10, 200, n),
        "Failure": np.random.choice([0, 1], n, p=[0.9, 0.1])
    }
    return pd.DataFrame(data)

# Train ML Model
def train_model(df):
    X = df[["Temperature", "Vibration", "Pressure"]]
    y = df["Failure"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy

# Streamlit UI
st.title("AI-Powered Predictive Maintenance System")
st.sidebar.header("Settings")
num_sensors = st.sidebar.slider("Number of Sensor Data Points", 50, 500, 100)

st.subheader("Real-Time Sensor Data")
data = generate_sensor_data(num_sensors)
st.write(data.head())

# Train Model
model, acc = train_model(data)
st.subheader("Model Performance")
st.write(f"Accuracy: {acc:.2f}")

# Predict Failures
st.subheader("Predict Maintenance Needs")
input_data = {
    "Temperature": st.number_input("Temperature", 20.0, 100.0, 50.0),
    "Vibration": st.number_input("Vibration", 0.1, 5.0, 2.0),
    "Pressure": st.number_input("Pressure", 10.0, 200.0, 100.0)
}
input_df = pd.DataFrame([input_data])
prediction = model.predict(input_df)

if prediction[0] == 1:
    st.error("⚠️ Potential failure detected! Schedule maintenance immediately.")
else:
    st.success("✅ No issues detected. System is running smoothly.")

st.subheader("System Insights")
st.write("- IoT Sensors continuously monitor key parameters.")
st.write("- Machine Learning predicts failures based on real-time data.")
st.write("- Dashboard provides centralized monitoring for proactive maintenance.")

st.sidebar.markdown("---")
st.sidebar.write("Developed by the solvers")
