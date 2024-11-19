import streamlit as st
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Function to load the trained model
def load_model():
    model = joblib.load(open('voting_model.pkl', 'rb'))
    return model

def load_scaler():
    scaler = joblib.load(open('scaler.pkl', 'rb'))
    return scaler

# Function to categorize blood pressure
def bp_category(ap_hi, ap_lo):
    if ap_hi < 120 and ap_lo < 80:
        return 'Normal'
    elif 120 <= ap_hi <= 129 and ap_lo < 80:
        return 'Elevated'
    elif 130 <= ap_hi <= 139 or 80 <= ap_lo <= 89:
        return 'Hypertension Stage 1'
    elif 140 <= ap_hi <= 180 or 90 <= ap_lo <= 120:
        return 'Hypertension Stage 2'
    else:
        return 'Hypertensive Crisis'

# Function to categorize cholesterol level
def cholesterol_category(cholesterol):
    if cholesterol < 200:
        return 'Normal'
    elif 200 <= cholesterol < 240:
        return 'Above normal'
    else:
        return 'High'

# Function to categorize glucose level
def glucose_category(glucose):
    if glucose < 100:
        return 'Normal'
    elif 100 <= glucose < 126:
        return 'Impaired Fasting Glucose'
    else:
        return 'Diabetes/Hyperglycemia'

# Function to collect user input
def get_user_input():
    st.title("Cardiovascular Disease Risk Prediction")

    age = st.number_input("Age", min_value=0, max_value=150)
    gender = st.radio("Gender", ("Male", "Female"))
    height = st.number_input("Height (cm)", min_value=0, max_value=300)
    weight = st.number_input("Weight (kg)", min_value=0, max_value=200)
    ap_hi = st.number_input("Systolic Blood Pressure (ap_hi)", min_value=0, max_value=300)
    ap_lo = st.number_input("Diastolic Blood Pressure (ap_lo)", min_value=0, max_value=200)
    
    cholesterol = st.number_input("Cholesterol level", min_value=0, max_value=300, help="Enter cholesterol level (e.g., 120)")
    gluc = st.number_input("Glucose level", min_value=0, max_value=300, help="Enter glucose level (e.g., 90)")

    smoke = st.radio("Do you smoke?", ("Yes", "No"))
    alco = st.radio("Do you consume alcohol?", ("Yes", "No"))
    active = st.radio("Are you physically active?", ("Yes", "No"))

    # BMI calculation (if not input directly)
    if height > 0:
        bmi = weight / (height / 100) ** 2
    else:
        bmi = 0 

    return age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, bmi

# Define the preprocess_input function
def preprocess_input(age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, bmi):
    # Map gender to numeric
    gender = 1 if gender == "Male" else 0
    
    # Use the cholesterol and glucose categories (as strings, no need to create binary columns)
    cholesterol_cat = cholesterol_category(cholesterol)
    glucose_cat = glucose_category(gluc)
    
    # Map smoke, alcohol, and activity levels
    smoke = 1 if smoke == "Yes" else 0
    alco = 1 if alco == "Yes" else 0
    active = 1 if active == "Yes" else 0
    
    # BMI calculation is already done, so we include it directly
    bmi = bmi
    
    # Prepare data as a dictionary (no new features)
    data = {
        'age': age,
        'gender': gender,
        'height': height,
        'weight': weight,
        'ap_hi': ap_hi,
        'ap_lo': ap_lo,
        'cholesterol': cholesterol,
        'gluc': gluc,
        'smoke': smoke,
        'alco': alco,
        'active': active,
        'BMI': bmi,
        'cholesterol_cat': cholesterol_cat,  # Keep the category as a string for input
        'glucose_cat': glucose_cat          # Keep the category as a string for input
    }
    
    # Convert to DataFrame
    df = pd.DataFrame([data])

    # Encode categorical columns (cholesterol_cat, glucose_cat)
    df = pd.get_dummies(df, columns=['cholesterol_cat', 'glucose_cat'], drop_first=True)

    df['bp_cat'] = df.apply(lambda x: bp_category(x['ap_hi'], x['ap_lo']), axis=1)

    df['bp_category_Hypertension Stage 1'] = df['bp_cat'] == 'Hypertension Stage 1'
    df['bp_category_Hypertension Stage 2'] = df['bp_cat'] == 'Hypertension Stage 2'
    df['bp_category_Hypertensive Crisis'] = df['bp_cat'] == 'Hypertensive Crisis'
    df['bp_category_Normal'] = df['bp_cat'] == 'Normal'

    categorical_cols = ['bp_cat']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df

# Function to make prediction
def make_prediction(input_data, model):
    # Check if the model has the method `predict_proba`
    if hasattr(model, "predict_proba"):
        # Get prediction probabilities (returns probabilities for both classes)
        prediction_proba = model.predict_proba(input_data)
        # We are interested in the probability for 'High risk' (class 1)
        risk_probability = prediction_proba[0][1]  # Probability for high risk (1)
        # Get the class prediction (high or low risk)
        prediction = model.predict(input_data)
    else:
        # If the model doesn't support `predict_proba`, just return prediction
        prediction = model.predict(input_data)
        risk_probability = None  # No probability available

    return prediction, risk_probability

def display_result(prediction, risk_probability):
    # Convert prediction to human-readable format
    result = "High risk" if prediction == 1 else "Low risk"
    # Display prediction and confidence score
    st.write(f"Prediction: {result}")
    
    if risk_probability is not None:
        st.write(f"Risk Probability: {risk_probability * 100:.2f}%")
    else:
        st.write("Risk probability not available for this model.")


# Main function to run the Streamlit app
def main():
    model = load_model()

    # Get user input
    age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, bmi = get_user_input()

    # Preprocess the input data
    input_data = preprocess_input(age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, bmi)

    scaler = load_scaler()

    # Scale input data
    scaled_df = scaler.transform(input_data)

    if st.button("Predict"):
        # Get prediction and risk probability
        prediction, risk_probability = make_prediction(scaled_df, model)

        # Display the result with risk probability
        display_result(prediction, risk_probability)

main()