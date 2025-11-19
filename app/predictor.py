import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os
import json

# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Currency conversion rate (EUR to INR)
# As of November 2025, 1 EUR ≈ 90 INR (approximate rate)
EUR_TO_INR_RATE = 90.0

# Load the trained model and encoders
@st.cache_resource
def load_model_and_encoders():
    """Load the trained model and encoders"""
    try:
        # Load the best model (you can change this to any model you prefer)
        model_path = os.path.join(project_root, "models", "random_forest.pkl")
        model = joblib.load(model_path)
        
        # Load encoders
        encoders = {
            'city': joblib.load(os.path.join(project_root, "models", "city_encoder.pkl")) if os.path.exists(os.path.join(project_root, "models", "city_encoder.pkl")) else None,
            'room_type': joblib.load(os.path.join(project_root, "models", "room_type_encoder.pkl")) if os.path.exists(os.path.join(project_root, "models", "room_type_encoder.pkl")) else None,
            'season': joblib.load(os.path.join(project_root, "models", "season_encoder.pkl")) if os.path.exists(os.path.join(project_root, "models", "season_encoder.pkl")) else None,
            'bathroom_type': joblib.load(os.path.join(project_root, "models", "bathroom_type_encoder.pkl")) if os.path.exists(os.path.join(project_root, "models", "bathroom_type_encoder.pkl")) else None
        }
        
        # Load scaler
        scaler = joblib.load(os.path.join(project_root, "models", "scaler.pkl")) if os.path.exists(os.path.join(project_root, "models", "scaler.pkl")) else None
        
        return model, encoders, scaler
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, {}, None

# Load model metrics
@st.cache_resource
def load_model_metrics():
    """Load model performance metrics"""
    try:
        metrics_path = os.path.join(project_root, "models", "model_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            return metrics
        else:
            return None
    except Exception as e:
        st.warning(f"Could not load model metrics: {str(e)}")
        return None

def preprocess_input(user_inputs, encoders, scaler):
    """
    Preprocess user inputs for prediction
    
    Args:
        user_inputs (dict): Dictionary of user inputs
        encoders (dict): Dictionary of label encoders
        scaler: Scaler object
        
    Returns:
        pd.DataFrame: Processed features ready for prediction
    """
    # Create a dataframe with user inputs
    df = pd.DataFrame([user_inputs])
    
    # Encode categorical variables
    categorical_cols = ['city', 'room_type', 'season', 'bathroom_type']
    for col in categorical_cols:
        if col in df.columns and encoders.get(col) is not None:
            try:
                df[f'{col}_encoded'] = encoders[col].transform(df[col])
            except ValueError:
                # Handle unseen labels
                df[f'{col}_encoded'] = 0
    
    # Convert season to numeric factor
    season_mapping = {'low': 1, 'shoulder': 2, 'peak': 3}
    df['season_factor'] = df['season'].map(season_mapping)
    
    # Create interaction features
    df['rating_num_reviews'] = df['rating'] * df['num_reviews']
    df['density'] = df['beds_in_room'] / df['room_area_sqft']
    df['distance_season'] = df['distance_to_center_km'] * df['season_factor']
    df['rating_occupancy'] = df['rating'] * df['occupancy_rate']
    
    # Create demand index
    # For simplicity, we'll use fixed values for normalization
    df['num_recent_reviews'] = (df['num_reviews'] - 65) / (320 - 65)  # Normalized between min and max
    
    # Calculate demand index
    df['demand_index'] = (
        0.4 * df['occupancy_rate'] +
        0.3 * df['num_recent_reviews'] +
        0.3 * df['season_factor'] / 3  # Normalize season factor
    )
    
    # Select features for modeling
    feature_columns = [
        'distance_to_center_km', 'rating', 'num_reviews', 'beds_in_room',
        'breakfast_included', 'wifi', 'laundry', 'weekend', 'occupancy_rate',
        'hostel_age_years', 'kitchen_access', 'air_conditioning', 
        'locker_available', 'security_24h', 'female_only_dorm_available',
        'common_area_count', 'room_area_sqft', 'city_encoded', 
        'room_type_encoded', 'season_encoded', 'bathroom_type_encoded',
        'rating_num_reviews', 'density', 'distance_season', 
        'rating_occupancy', 'demand_index'
    ]
    
    # Ensure all features are present
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    X = df[feature_columns]
    
    # Clamp negative values to zero
    X = X.clip(lower=0)
    
    # Scale features if scaler is available
    if scaler is not None:
        X_scaled = scaler.transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)
        # Clamp negative values to zero after scaling
        X = X.clip(lower=0)
    
    return X

def display_model_metrics(metrics):
    """Display model performance metrics in the Streamlit app"""
    if metrics is None:
        st.warning("Model metrics not available.")
        return
    
    st.subheader("📊 Model Performance Metrics")
    
    # Create tabs for different models
    model_tabs = st.tabs(list(metrics.keys()))
    
    for i, (model_name, model_metrics) in enumerate(metrics.items()):
        with model_tabs[i]:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Convert MAE to INR
                mae_inr = model_metrics['MAE'] * EUR_TO_INR_RATE
                st.metric("Mean Absolute Error (MAE)", f"₹{mae_inr:.2f}")
            
            with col2:
                # Convert RMSE to INR
                rmse_inr = model_metrics['RMSE'] * EUR_TO_INR_RATE
                st.metric("Root Mean Squared Error (RMSE)", f"₹{rmse_inr:.2f}")
            
            with col3:
                st.metric("R² Score", f"{model_metrics['R2']:.3f}")
            
            # Show model name
            st.caption(f"*Metrics for {model_name} model*")

def main():
    st.set_page_config(page_title="Hostel Price Predictor", page_icon="🏨", layout="wide")
    
    st.title("🏨 Hostel Price Prediction")
    st.markdown("""
    This application predicts the price per night for hostel beds based on various factors.
    Enter the details of the hostel and room below to get a price prediction.
    """)
    
    # Add currency toggle
    currency_option = st.radio("Select Currency", ["Indian Rupees (₹)", "Euros (€)"], index=0)
    use_inr = (currency_option == "Indian Rupees (₹)")
    
    # Load model and encoders
    model, encoders, scaler = load_model_and_encoders()
    
    # Load model metrics
    metrics = load_model_metrics()
    
    if model is None:
        st.error("Failed to load the prediction model. Please check the model files.")
        return
    
    # Display model metrics
    display_model_metrics(metrics)
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Location & Basic Information")
        city = st.selectbox("City", [
            "Berlin", "Paris", "London", "Amsterdam", "Barcelona", 
            "Rome", "Prague", "Vienna", "Budapest", "Dublin",
            "Madrid", "Athens", "Lisbon", "Warsaw", "Stockholm",
            "Copenhagen", "Oslo", "Helsinki", "Brussels", "Milan",
            # Indian cities
            "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai",
            "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Goa",
            "Kochi", "Mysore", "Darjeeling", "Shimla", "Manali"
        ])
        
        distance_to_center = st.slider("Distance to City Center (km)", 0.0, 10.0, 1.0, 0.1)
        rating = st.slider("Hostel Rating", 1.0, 5.0, 4.0, 0.1)
        num_reviews = st.number_input("Number of Reviews", 0, 1000, 100)
        
        room_type = st.radio("Room Type", ["dorm", "private"])
        beds_in_room = st.number_input("Beds in Room", 1, 20, 4)
        room_area = st.number_input("Room Area (sq ft)", 50, 500, 150)
        
    with col2:
        st.subheader("Amenities & Features")
        col2a, col2b = st.columns(2)
        
        with col2a:
            breakfast = st.checkbox("Breakfast Included", value=True)
            wifi = st.checkbox("WiFi Available", value=True)
            laundry = st.checkbox("Laundry Service", value=True)
            kitchen = st.checkbox("Kitchen Access", value=True)
            ac = st.checkbox("Air Conditioning", value=True)
            
        with col2b:
            locker = st.checkbox("Locker Available", value=True)
            security = st.checkbox("24h Security", value=True)
            female_dorm = st.checkbox("Female-only Dorm", value=False)
            common_areas = st.number_input("Common Areas Count", 0, 10, 2)
            
        st.subheader("Booking Information")
        season = st.selectbox("Travel Season", ["low", "shoulder", "peak"])
        weekend = st.checkbox("Weekend Stay", value=False)
        occupancy_rate = st.slider("Occupancy Rate", 0.0, 1.0, 0.7, 0.05)
        hostel_age = st.number_input("Hostel Age (years)", 0, 50, 5)
        bathroom_type = st.selectbox("Bathroom Type", ["shared", "private"])
    
    # Prediction button
    if st.button("🔮 Predict Price", type="primary", use_container_width=True):
        # Collect all inputs
        user_inputs = {
            'city': city,
            'distance_to_center_km': distance_to_center,
            'rating': rating,
            'num_reviews': num_reviews,
            'room_type': room_type,
            'beds_in_room': beds_in_room,
            'room_area_sqft': room_area,
            'breakfast_included': int(breakfast),
            'wifi': int(wifi),
            'laundry': int(laundry),
            'kitchen_access': int(kitchen),
            'air_conditioning': int(ac),
            'locker_available': int(locker),
            'security_24h': int(security),
            'female_only_dorm_available': int(female_dorm),
            'common_area_count': common_areas,
            'season': season,
            'weekend': int(weekend),
            'occupancy_rate': occupancy_rate,
            'hostel_age_years': hostel_age,
            'bathroom_type': bathroom_type
        }
        
        # Preprocess inputs
        try:
            processed_features = preprocess_input(user_inputs, encoders, scaler)
            
            # Make prediction
            prediction = model.predict(processed_features)[0]
            
            # Clamp prediction to zero if negative
            prediction = max(0, prediction)
            
            # Display result in selected currency
            if use_inr:
                prediction_inr = prediction * EUR_TO_INR_RATE
                st.success(f"💰 Predicted Price: ₹{prediction_inr:.2f} per night")
            else:
                st.success(f"💰 Predicted Price: €{prediction:.2f} per night")
            
            # Show feature importance
            st.subheader("🔍 Key factors")
            feature_importance = pd.DataFrame({
                'Feature': processed_features.columns,
                'Value': processed_features.iloc[0]
            }).sort_values('Value', key=abs, ascending=False).head(10)
            
            st.dataframe(feature_importance)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    
    # Information section
    st.markdown("---")
    st.subheader("About this App")
    st.markdown("""
    This application uses machine learning to predict hostel prices based on:
    - **Location factors**: City, distance to center
    - **Property features**: Age, amenities, security
    - **Room characteristics**: Type, beds, area
    - **Demand indicators**: Occupancy, reviews, season
    - **Engineered features**: Demand index, interaction terms
    
    The model was trained on hostel data from major European and Indian cities and achieved high accuracy 
    in predicting prices across different scenarios.
    """)

if __name__ == "__main__":
    main()