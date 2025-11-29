import tensorflow as tf
import pandas as pd
import streamlit as st
import numpy as np
import pickle

# Load model safely
model = tf.keras.models.load_model('RegressionModel.h5', compile=False)

# Load preprocessing objects
with open('LabelEncoder.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('OneHotEncoder.pkl', 'rb') as file:
    one_hot_encoder_geo = pickle.load(file)

with open('StandardScaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.title("Customer Salary Predictor")

# User input fields
geography = st.selectbox('Geography', one_hot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
Exited=st.selectbox('Exited',[0,1])

# DataFrame preparation
df = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited':[Exited]
})

# Encode geography (ONLY transform, do not fit again!)
geo_encoded = one_hot_encoder_geo.transform([[geography]]).toarray()
geo_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))

# Combine dataframe
final_data = pd.concat([df, geo_df], axis=1)

# Scaling (ONLY transform)
final_scaled = scaler.transform(final_data)

# Prediction
prediction = model.predict(final_scaled)[0][0]

st.subheader("Predicted Salary")
st.write(f"₹ {prediction:,.2f}")
