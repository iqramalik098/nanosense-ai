import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Page settings
st.set_page_config(
    page_title="NanoSense AI",
    page_icon="🧬"
)

# Application title
st.title("🧬 NanoSense AI")

st.subheader(
    "AI-Powered Nanobiosensor Simulator for Healthcare"
)

st.write(
    "Explore how machine learning can analyze "
    "simulated nanobiosensor measurements."
)

st.warning(
    "Educational prototype only. "
    "This application uses simulated data "
    "and cannot diagnose disease."
)

# Generate simulated data
np.random.seed(42)

concentration = np.linspace(0, 100, 200)

sensor_response = (
    5 * concentration / (25 + concentration)
    + np.random.normal(0, 0.08, 200)
)

data = pd.DataFrame({
    "Concentration": concentration,
    "Sensor_Response": sensor_response
})

# Prepare machine learning data
X = data[["Concentration"]]
y = data["Sensor_Response"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train the AI model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# User interaction
st.header("🔬 Sensor Simulation")

selected_concentration = st.slider(
    "Select simulated concentration",
    min_value=0,
    max_value=100,
    value=50
)

# Predict sensor response
prediction = model.predict(
    pd.DataFrame({
        "Concentration": [selected_concentration]
    })
)[0]

st.metric(
    "Predicted sensor response",
    f"{prediction:.3f} arbitrary units"
)

# Display data visualization
st.header("📊 Sensor Response Graph")

fig, ax = plt.subplots(figsize=(8, 4))

ax.scatter(
    data["Concentration"],
    data["Sensor_Response"],
    alpha=0.6,
    label="Simulated measurements"
)

ax.set_xlabel("Concentration (arbitrary units)")
ax.set_ylabel("Sensor response (arbitrary units)")
ax.set_title("Simulated Nanobiosensor Response")
ax.legend()
ax.grid(True, alpha=0.3)

st.pyplot(fig)

# Evaluate the model
test_predictions = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    test_predictions
)

st.header("🤖 Model Evaluation")

st.metric(
    "Mean Absolute Error",
    f"{mae:.3f}"
)

st.write(
    "The model is evaluated on held-out synthetic data. "
    "These results demonstrate a machine-learning workflow "
    "and do not establish real-world sensor performance."
)

# Scientific explanation
st.header("🧪 Scientific Background")

st.write(
    "Nanomaterials are investigated for biosensing because "
    "their physical and chemical properties can support "
    "the detection of biological signals."
)

st.write(
    "This prototype uses a simplified mathematical "
    "relationship to generate synthetic measurements. "
    "The Random Forest model learns patterns in the "
    "simulated data."
)

st.subheader("Limitations")

st.write(
    "No experimental or patient data are used. "
    "This application is not a validated medical device "
    "and must not be used for diagnosis or treatment."
)
