import streamlit as st
import numpy as np
import tensorflow as tf
import pickle

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="model_energy.tflite")
interpreter.allocate_tensors()

# Load scaler dan label encoder
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# Fungsi prediksi menggunakan model TFLite
def predict_energy(input_data):
    input_data = scaler.transform([input_data])
    input_data = np.array(input_data, dtype=np.float32)

    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_index)
    return prediction[0][0]

# Streamlit UI
st.set_page_config(page_title="Energy Consumption Prediction", layout="centered")

st.title("🔋 Energy Consumption Prediction")
st.markdown("Masukkan data berikut untuk memprediksi konsumsi energi:")

# Input fields
temperature = st.number_input("🌡️ Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
humidity = st.slider("💧 Humidity (%)", min_value=0, max_value=100, value=50)
square_footage = st.number_input("🏢 Square Footage", min_value=0.0, value=1500.0)
occupancy = st.number_input("👥 Occupancy", min_value=0, value=5)
renewable_energy = st.number_input("☀️ Renewable Energy (kWh)", min_value=0.0, value=100.0)

day_of_week = st.selectbox("📅 Day of Week", label_encoders["DayOfWeek"].classes_)
hvac_usage = st.selectbox("❄️ HVAC Usage", label_encoders["HVACUsage"].classes_)
lighting_usage = st.selectbox("💡 Lighting Usage", label_encoders["LightingUsage"].classes_)
holiday = st.selectbox("🎉 Holiday", label_encoders["Holiday"].classes_)

# Encode categorical input
day_of_week_encoded = label_encoders["DayOfWeek"].transform([day_of_week])[0]
hvac_encoded = label_encoders["HVACUsage"].transform([hvac_usage])[0]
lighting_encoded = label_encoders["LightingUsage"].transform([lighting_usage])[0]
holiday_encoded = label_encoders["Holiday"].transform([holiday])[0]

# Buat input array SESUAI dengan urutan saat training
input_data = [
    temperature, humidity, square_footage, occupancy,
    renewable_energy, day_of_week_encoded,
    hvac_encoded, lighting_encoded, holiday_encoded
]

# Prediksi saat tombol ditekan
if st.button("🔮 Prediksi Konsumsi Energi"):
    prediction = predict_energy(input_data)
    st.success(f"Prediksi Konsumsi Energi: {prediction:.2f} kWh")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit & TensorFlow Lite")
