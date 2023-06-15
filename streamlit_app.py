import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, QuantileTransformer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import pickle
import streamlit as st
import time
import os

st.title('House price predictor')
st.markdown('> 21KDL - Introduction to Data Science')

Area = st.number_input("Choose your area")
Bedroom	= st.number_input("Number of bedrooms")
Bathroom = st.number_input("Number of bathrooms")
Frontage = st.number_input("Number of frontages")
Floors = st.number_input("Number of floors")

model = pickle.load(open("stacking_model.sav", 'rb'))

df = pd.read_csv('labeled_data.csv', index_col= [0])
locs = df.columns[5:]
Loc = st.selectbox("Select your location", tuple(locs))

input = np.array([np.zeros(80, np.float32)])
input[0][0] = Area
input[0][1] = Bedroom
input[0][2] = Bathroom
input[0][3] = Frontage
input[0][4] = Floors
cols = list(df.columns)
input[0][cols.index(Loc)] = 1

if st.button('Predict'):
    predict_price = pickle.load(open("stacking_model.sav", 'rb')).predict(input)
    st.write(f"Predicted house price: {predict_price[0]}")
else:
    st.write('')
