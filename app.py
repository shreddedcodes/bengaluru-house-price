import streamlit as st
import joblib
import numpy as np
import pandas as pd

le_area = joblib.load('le_area.pkl')
le_location = joblib.load('le_location.pkl')
scaler_X_std = joblib.load('scaler_X_std.pkl')
scaler_X_minmax = joblib.load('scaler_X_minmax.pkl')
scaler_y_minmax = joblib.load('scaler_y_minmax.pkl')
model = joblib.load('xgb_model.pkl')

def encode_avail(val):
    if pd.isnull(val):
        return np.nan
    val = val.strip().lower()
    if val in ['ready to move', 'immediate possession']:
        return 0
    return 1

st.title("🏠 Bangalore House Price Predictor")
st.markdown("Enter the property details to predict the price.")

area = st.selectbox("Area", le_area.classes_)
location = st.selectbox("Location", le_location.classes_)
availability = st.selectbox("Availability", ['Ready to move', 'Immediate possession', 'Under Construction'])
size = st.number_input("BHK Size (e.g. 2, 3, 4)", min_value=1, max_value=10, step=1)
total_sqft = st.number_input("Total Square Feet", min_value=200.0, step=50.0)
bath = st.number_input("Number of Bathrooms", min_value=1, max_value=10, step=1)
balcony = st.number_input("Number of Balconies", min_value=0, max_value=5, step=1)

if st.button("Predict"):
    if bath > size:
        st.error("🚫 Number of bathrooms cannot be greater than the number of rooms (BHK). Please correct the input.")
    else:
        area_enc = le_area.transform([area])[0]
        location_enc = le_location.transform([location])[0]
        availability_enc = encode_avail(availability)
    
        raw_input = np.array([[area_enc, location_enc, availability_enc, size, total_sqft, bath, balcony]])
    
        scaled_input_std = scaler_X_std.transform(raw_input)
        scaled_input_minmax = scaler_X_minmax.transform(scaled_input_std)
    
        price_pred = model.predict(scaled_input_minmax)
        actual_price = scaler_y_minmax.inverse_transform(price_pred.reshape(-1, 1))[0][0]
    
        #st.write("Raw model prediction (scaled):", price_pred)
        #st.write("Final actual price:", actual_price)
    
        st.success(f"🏠 Predicted House Price: ₹{actual_price:,.2f}")
