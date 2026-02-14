import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Application Setup
st.set_page_config(page_title="Egypt Real Estate Price Predictor", page_icon="🏠")

st.title("Egypt Real Estate Price Predictor 🏠")
st.write("Enter the property details to get an estimated market price.")

# Load the pre-trained Pipeline
@st.cache_resource
def load_model():
    # Ensure the file 'house_price_pipeline.joblib' is in the same directory
    return joblib.load('house_price_pipeline.joblib')

try:
    pipeline = load_model()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Layout: Two columns for input fields
col1, col2 = st.columns(2)

with col1:
    city = st.selectbox("City", 
                        ['Nasr City', 'New Cairo - El Tagamoa', 'Sheikh Zayed', 
                         '6th of October', 'Maadi', 'Rehab City'])
    
    unit_type = st.selectbox("Unit Type", 
                             ['Apartment', 'Duplex', 'Chalet', 'Stand Alone Villa', 'Town House'])
    
    area = st.number_input("Area (sqm)", min_value=10, max_value=2000, value=150)

with col2:
    bedrooms = st.slider("Number of Bedrooms", 1, 10, 3)
    bathrooms = st.slider("Number of Bathrooms", 1, 10, 2)
    payment = st.selectbox("Payment Option", 
                           ['Cash', 'Installment', 'Cash or Installment'])
    
    delivery = st.selectbox("Delivery Term", 
                            ['Finished', 'Semi Finished', 'Core & Shell', 'Not Finished'])

# Prediction Button
if st.button("Predict Estimated Price"):
    # Prepare data for the model
    # Note: Column names must exactly match the ones used during model training
    input_data = pd.DataFrame({
        'Type': [unit_type],
        'Bedrooms': [bedrooms],
        'Bathrooms': [bathrooms],
        'Area': [area],
        'Payment_Option': [payment],
        'Delivery_Term': [delivery],
        'City': [city]
    })
    
    try:
        # Perform prediction using the Pipeline
        prediction = pipeline.predict(input_data)[0]
        
        # Display the result
        st.success(f"The estimated price is: EGP {prediction:,.2f}")
        st.balloons()
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# Footer
st.markdown("---")
st.caption("Data source: Based on historical real estate market listings in Egypt.")