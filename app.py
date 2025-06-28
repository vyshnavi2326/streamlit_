import streamlit as st

st.title("💼 Employee Salary Estimator")

st.markdown("This is a simple rule-based app to estimate salary based on experience, education, and job role.")

# Sidebar inputs
experience = st.slider("Years of Experience", 0, 20, 2)
education = st.selectbox("Education Level", ["Bachelors", "Masters", "PhD"])
role = st.selectbox("Job Role", ["Analyst", "Engineer", "Manager"])
location = st.selectbox("Location", ["Hyderabad", "Bangalore", "Chennai", "Mumbai", "Delhi"])

# Basic rule-based logic
base_salary = 20000
experience_bonus = experience * 2000

if education == "Masters":
    edu_bonus = 10000
elif education == "PhD":
    edu_bonus = 20000
else:
    edu_bonus = 0

if role == "Engineer":
    role_bonus = 15000
elif role == "Manager":
    role_bonus = 25000
else:
    role_bonus = 10000

location_bonus = {
    "Hyderabad": 3000,
    "Bangalore": 8000,
    "Chennai": 4000,
    "Mumbai": 7000,
    "Delhi": 5000
}[location]

# Final salary
predicted_salary = base_salary + experience_bonus + edu_bonus + role_bonus + location_bonus

# Output
if st.button("Predict Salary"):
    st.success(f"Estimated Salary: ₹ {predicted_salary:,}")
