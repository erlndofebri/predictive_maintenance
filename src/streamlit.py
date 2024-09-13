# Import library
import streamlit as st
import requests
import pandas as pd


# Buat title
st.title("Machine Maintenance Prediction")
#st.subheader("Enter some variables and click Predict button")


# Create form of input
with st.form(key = "machine_data_form"):
    # Buat data Type, hanya L, M, dan H (radio button)
    types = st.radio(
        label = "1. \tMasukkan tipe kualitas produksi",
        options = ["L", "M", "H"],
        captions = ["Low", "Medium", "High"],
        index = 0,
        horizontal = True
    )

    # Buat data Air temperature
    air_temp = st.number_input(
        label = "2.\tMasukkan Air Temperature [K]:",
        min_value = 273.15,
        max_value = 373.15,
        help = "Value range from 273.15 to 373.15"
    )

    # Buat data Process temperature
    process_temp = st.number_input(
        label = "3.\tMasukkan Process Temperature [K]:",
        min_value = 273.15,
        max_value = 373.15,
        help = "Value range from 273.15 to 373.15"
    )

    # Buat data Rotational Speed
    rot_speed = st.number_input(
        label = "4.\tMasukkan Tools Rotational Speed [rpm]:",
        min_value = 0.0,
        max_value = 3000.0,
        help = "Value range from 0.0 to 3,000.0"
    )

    # Buat data Torque
    torque = st.number_input(
        label = "5.\tMasukkan Torque [Nm]:",
        min_value = 0.0,
        max_value = 100.0,
        help = "Value range from 0.0 to 100.0"
    )

    # Buat data Tool wear
    tool_wear = st.number_input(
        label = "6.\tMasukkan Tool wear [min]:",
        min_value = 0.0,
        max_value = 500.0,
        help = "Value range from 0 to 500"
    )

    # Buat submit button
    submitted = st.form_submit_button("Predict")

    # Buat kondisi ketika submit
    if submitted:
        # Buat data
        machine_data = {
            "types": types,
            "air_temperature": air_temp,
            "process_temperature": process_temp,
            "rotational_speed": rot_speed,
            "torque": torque,
            "tool_wear": tool_wear
        }

        # Buat loading animation untuk kirimkan data
        with st.spinner("Kirim data untuk diprediksi server ..."):
            res = requests.post("http://localhost:8000/predict",
                                json = machine_data).json()
            
        # Tampilkan hasil prediksi
        if res["error_msg"] != "":
            st.error("Error terjadi ketika melakukan prediksi")
        else:
            # Edit sedikit output
            res_pred = "Failure" if res['res']==1 else "No Failure"

            # Tampilkan hasil
            st.success(f"""
                Sukses!  
                Hasil prediksi: **{res_pred}**  
                Probability failure: **{res['res_proba']}**
            """)
