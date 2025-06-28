import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
@st.cache_data
def load_data():
    # Sample data (replace with your real dataset)
    data = pd.DataFrame({
        'Experience': [1, 3, 5, 7, 9, 11, 13],
        'Education': ['Bachelors', 'Masters', 'Bachelors', 'PhD', 'Masters', 'PhD', 'Bachelors'],
        'Role': ['Analyst', 'Engineer', 'Manager', 'Analyst', 'Manager', 'Engineer', 'Analyst'],
        'Location': ['Hyderabad', 'Bangalore', 'Chennai', 'Delhi', 'Mumbai', 'Hyderabad', 'Delhi'],
        'Salary': [30000, 45000, 60000, 70000, 90000, 85000, 50000]
    })
    return data

# Preprocess the data
def preprocess_data(df):
    df_encoded = df.copy()
    label_encoders = {}
    for column in ['Education', 'Role', 'Location']:
        le = LabelEncoder()
        df_encoded[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    return df_encoded, label_encoders

# Train model
def train_model(df):
    X = df.drop('Salary', axis=1)
    y = df['Salary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Streamlit app
st.title("Employee Salary Prediction")

data = load_data()
st.subheader("Dataset")
st.dataframe(data)

# Preprocess and train
df_encoded, encoders = preprocess_data(data)
model = train_model(df_encoded)

# User input
st.sidebar.header("Enter Employee Details")
experience = st.sidebar.slider("Years of Experience", 0, 20, 1)
education = st.sidebar.selectbox("Education Level", data['Education'].unique())
role = st.sidebar.selectbox("Job Role", data['Role'].unique())
location = st.sidebar.selectbox("Location", data['Location'].unique())

# Encode input
input_data = pd.DataFrame({
    'Experience': [experience],
    'Education': [encoders['Education'].transform([education])[0]],
    'Role': [encoders['Role'].transform([role])[0]],
    'Location': [encoders['Location'].transform([location])[0]]
})

# Predict
if st.sidebar.button("Predict Salary"):
    prediction = model.predict(input_data)
    st.success(f"Estimated Salary: ₹ {int(prediction[0]):,}")

# Visualization
st.subheader("Salary Distribution")
fig, ax = plt.subplots()
sns.histplot(data['Salary'], kde=True, ax=ax)
st.pyplot(fig)
