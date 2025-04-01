import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle
import tensorflow as tf


##Load the trained model
model = tf.keras.models.load_model(r'E:\Desktop\sadia waheed GU1\Gen AI course 4 months\ANN-1P\env\model.h5')


## load the encoder scaler
with open('E:\\Desktop\sadia waheed GU1\Gen AI course 4 months\ANN-1P\env\label_gender.pkl', 'rb') as file:
    label_gender = pickle.load(file)
with open('E:\\Desktop\sadia waheed GU1\Gen AI course 4 months\ANN-1P\env\one_hot_encoder_geo.pkl', 'rb') as file:
    one_hot_encoder_geo = pickle.load(file)
with open('E:\\Desktop\sadia waheed GU1\Gen AI course 4 months\ANN-1P\env\scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

##streamlit app
st.title("Customer Retension Prediction")

##user input
geography = st.selectbox('Geography',one_hot_encoder_geo.categories_[0])
gender = st.selectbox('Gender',label_gender.classes_) 
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number Of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])

input_data = pd.DataFrame({
    'CreditScore' : [credit_score],
    'Gender' : [label_gender.transform([gender])[0]],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember' : [is_active_member],
    'EstimatedSalary' : [estimated_salary]
})

geo_encoded = one_hot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))

##combine one hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

#scale the input data
input_data_scaled = scaler.transform(input_data)

#predict Churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f"Customer Retension Probability: {prediction_proba: .2f}")

if prediction_proba > 0.5:
    print("The customer is likely to leave the bank")
else:
    print("The customer is not likely to leave the bank")



