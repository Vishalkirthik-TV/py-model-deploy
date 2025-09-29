import streamlit as st
import pandas as pd
import numpy as np
import joblib as jb

st.write("This is my first streamlit application")
energy = pd.read_csv(r"C:\Users\vkirt\Downloads\appliance_energy.csv")
st.line_chart(energy)

#load the model
my_model = jb.load(r"C:\Users\vkirt\Desktop\python learnings\EnergyLRmodel_1.pkl")

temp = st.number_input("Enter the temperature: ",min_value=0.0,max_value=150.0,value=10.0)

if st.button("Predict Energy Consumption"):
    data = np.array([[temp]])
    predictions = my_model.predict(data)
    st.write(f"The predicted energy consumption is {predictions[0]}")