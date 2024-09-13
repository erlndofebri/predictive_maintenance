import streamlit as st
from pages import predict, predict_dataset

# Add custom CSS to hide the page title if needed
hide_streamlit_style = """
    <style>
        .css-1r6slb7 {
            display: none;
        }
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Pilih Halaman", ["Predict Dataset", "Predict"])

# Render page based on sidebar selection
if page == "Predict Dataset":
    predict_dataset.predict_dataset()
elif page == "Predict":
    predict.predict()
