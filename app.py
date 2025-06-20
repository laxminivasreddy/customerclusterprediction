import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.cluster import KMeans
import streamlit as st

# load the model
kmeans = joblib.load("Model.pkl")
df = pd.read_csv("Mall_Customers.csv")
X = df[["Annual Income (k$)","Spending Score (1-100)"]]
X_array = X.values

# streamlit application page
st.set_page_config(page_title = "Customer Cluster Prediction", layout = "centered")
st.title("Customer Cluster Prediction" )
st.write("Enter the Customer Annual Income and Spending Score to predict the cluster")

# inputs

annual_income = st.number_input("annual income of a customer",min_value = 0,max_value = 400, value = 50)
spending_score = st.slider("spending score between 1-100",1,100,20)

#predict the cluster
if st.button("Predict Cluster"):
    input_data = np. array ([[annual_income, spending_score]])
    cluster = kmeans.predict(input_data)[0]
    st.success(f"Predicted Cluster is: {cluster}")