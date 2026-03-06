import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# -----------------------------
# Build model architecture
# -----------------------------
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(40,17)),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    return model


# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model():
    model = build_model()
    model.load_weights("turbofan_model.h5")
    return model


# -----------------------------
# Load scaler
# -----------------------------
@st.cache_resource
def load_scaler():
    scaler = joblib.load("scaler.pkl")
    return scaler


model = load_model()
scaler = load_scaler()


# -----------------------------
# Features used during training
# -----------------------------
feature_cols = [
    'operational_setting_1',
    'operational_setting_2',
    'sensor_2',
    'sensor_3',
    'sensor_4',
    'sensor_6',
    'sensor_7',
    'sensor_8',
    'sensor_9',
    'sensor_11',
    'sensor_12',
    'sensor_13',
    'sensor_14',
    'sensor_15',
    'sensor_17',
    'sensor_20',
    'sensor_21'
]


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Turbofan Engine Remaining Useful Life Prediction")

st.write("""
This application predicts the **Remaining Useful Life (RUL)** of a turbofan engine  
using a trained **LSTM deep learning model**.
""")

st.write("---")


# -----------------------------
# Upload CSV
# -----------------------------
uploaded_file = st.file_uploader("Upload Engine Sensor CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.write(df.head())

    missing = [col for col in feature_cols if col not in df.columns]

    if len(missing) > 0:
        st.error(f"Missing required columns: {missing}")

    else:

        sensor_data = df[feature_cols]

        scaled_data = scaler.transform(sensor_data)

        if len(scaled_data) < 40:
            st.error("Dataset must contain at least 40 cycles.")
        else:

            sequence = scaled_data[-40:]
            sequence = sequence.reshape(1,40,17)

            prediction = model.predict(sequence)

            rul = float(prediction[0][0])

            st.success(f"Predicted Remaining Useful Life: {round(rul,2)} cycles")

            # -----------------------------
            # Engine Health Status
            # -----------------------------
            st.subheader("Engine Health Status")

            if rul > 80:
                st.success("🟢 Engine Healthy")

            elif rul > 30:
                st.warning("🟡 Maintenance Required Soon")

            else:
                st.error("🔴 Critical: Engine Close to Failure")


# -----------------------------
# Example prediction
# -----------------------------
st.write("---")
st.subheader("Example Prediction")

if st.button("Run Example Prediction"):

    sample = np.zeros((1,40,17))

    prediction = model.predict(sample)

    rul = float(prediction[0][0])

    st.success(f"Predicted Remaining Useful Life: {round(rul,2)} cycles")


# -----------------------------
# Model Information
# -----------------------------
st.write("---")
st.subheader("Model Information")

st.write("Model Type: LSTM Neural Network")

st.write("Input Shape:")
st.code("(1, 40, 17)")

st.write("Output:")
st.code("Remaining Useful Life (cycles)")


# -----------------------------
# Footer
# -----------------------------
st.write("---")
st.write("Developed for Predictive Maintenance using NASA Turbofan Dataset")