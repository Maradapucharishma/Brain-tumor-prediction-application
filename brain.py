import streamlit as st
import numpy as np
import pandas as pd
import pickle

try:
    model = pickle.load(open(r"D:\streamlit\brainpredictor.pkl", "rb"))
except FileNotFoundError:
    st.error("Model file not found. Please make sure the file path is correct.")
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")



st.title("Brain Tumor Detection using ML")
area = st.number_input('Enter Area', value=0.0)
perimeter = st.number_input('Enter Perimeter', value=0.0)
solidity = st.number_input('Enter Solidity', value=0.0)
major_axis = st.number_input('Enter Major Axis', value=0.0)
minor_axis = st.number_input('Enter Minor Axis', value=0.0)
magnitude = st.number_input('Enter Magnitude', value=0.0)

model = pickle.load(open(r"D:\streamlit\brainpredictor.pkl",'rb'))
if st.button('Submit'):
    # Perform prediction using the model
    input_data = [[area, perimeter, solidity, major_axis, minor_axis, magnitude]]
    result = model.predict(input_data)
    st.write('Prediction:', result)

    if result == 1:
        st.write('Tumor')
    else:
        st.write('No Tumor')
