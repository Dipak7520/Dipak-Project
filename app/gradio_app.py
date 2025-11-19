import gradio as gr
import pandas as pd
import numpy as np
import joblib
import os
import json

# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Currency conversion rate (EUR to INR)
# As of November 2025, 1 EUR ≈ 90 INR (approximate rate)
EUR_TO_INR_RATE = 90.0

# Load the trained model
def load_model():
    """Load the trained model"""
    try:
        model_path = os.path.join(project_root, "models", "random_forest.pkl")
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

# Load model metrics
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
        print(f"Could not load model metrics: {str(e)}")
        return None

# Preprocessing function
def preprocess_inputs(inputs):
    """
    Preprocess inputs for prediction
    """
    # Create dataframe from inputs
    df = pd.DataFrame([inputs])
    
    # Simple encoding (in a real app, you would load the encoders)
    city_map = {
        "Berlin": 0, "Paris": 1, "London": 2, "Amsterdam": 3, "Barcelona": 4,
        "Rome": 5, "Prague": 6, "Vienna": 7, "Budapest": 8, "Dublin": 9,
        "Madrid": 10, "Athens": 11, "Lisbon": 12, "Warsaw": 13, "Stockholm": 14,
        "Copenhagen": 15, "Oslo": 16, "Helsinki": 17, "Brussels": 18, "Milan": 19,
        # Indian cities
        "Mumbai": 20, "Delhi": 21, "Bangalore": 22, "Hyderabad": 23, "Chennai": 24,
        "Kolkata": 25, "Pune": 26, "Ahmedabad": 27, "Jaipur": 28, "Goa": 29,
        "Kochi": 30, "Mysore": 31, "Darjeeling": 32, "Shimla": 33, "Manali": 34
    }
    
    room_type_map = {"dorm": 0, "private": 1}
    season_map = {"low": 1, "shoulder": 2, "peak": 3}
    bathroom_map = {"shared": 0, "private": 1}
    
    df['city_encoded'] = df['city'].map(city_map)
    df['room_type_encoded'] = df['room_type'].map(room_type_map)
    df['season_encoded'] = df['season'].map(season_map)
    df['bathroom_type_encoded'] = df['bathroom_type'].map(bathroom_map)
    
    # Convert season to numeric factor
    df['season_factor'] = df['season'].map(season_map)
    
    # Create interaction features
    df['rating_num_reviews'] = df['rating'] * df['num_reviews']
    df['density'] = df['beds_in_room'] / df['room_area_sqft']
    df['distance_season'] = df['distance_to_center_km'] * df['season_factor']
    df['rating_occupancy'] = df['rating'] * df['occupancy_rate']
    
    # Create demand index
    df['num_recent_reviews'] = (df['num_reviews'] - 65) / (320 - 65)  # Normalized
    
    # Calculate demand index
    df['demand_index'] = (
        0.4 * df['occupancy_rate'] +
        0.3 * df['num_recent_reviews'] +
        0.3 * df['season_factor'] / 3
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
    
    return X

# Format model metrics for display
def format_model_metrics(metrics, use_inr=False):
    """Format model metrics for display in the Gradio app"""
    if metrics is None:
        return "Model metrics not available."
    
    # Format metrics as a string
    metrics_text = "## 📊 Model Performance Metrics\n\n"
    
    for model_name, model_metrics in metrics.items():
        metrics_text += f"### {model_name}\n"
        if use_inr:
            mae_inr = model_metrics['MAE'] * EUR_TO_INR_RATE
            rmse_inr = model_metrics['RMSE'] * EUR_TO_INR_RATE
            metrics_text += f"- **Mean Absolute Error (MAE)**: ₹{mae_inr:.2f}\n"
            metrics_text += f"- **Root Mean Squared Error (RMSE)**: ₹{rmse_inr:.2f}\n"
        else:
            metrics_text += f"- **Mean Absolute Error (MAE)**: €{model_metrics['MAE']:.2f}\n"
            metrics_text += f"- **Root Mean Squared Error (RMSE)**: €{model_metrics['RMSE']:.2f}\n"
        metrics_text += f"- **R² Score**: {model_metrics['R2']:.3f}\n\n"
    
    return metrics_text

# Prediction function
def predict_price(
    city, distance_to_center, rating, num_reviews, room_type, beds_in_room,
    room_area, breakfast, wifi, laundry, kitchen, ac, locker, security,
    female_dorm, common_areas, season, weekend, occupancy_rate, hostel_age,
    bathroom_type, use_inr=False
):
    # Load model
    model = load_model()
    if model is None:
        return "Error: Could not load model", ""
    
    # Collect inputs
    inputs = {
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
        processed_features = preprocess_inputs(inputs)
        
        # Make prediction
        prediction = model.predict(processed_features)[0]
        
        # Clamp prediction to zero if negative
        prediction = max(0, prediction)
        
        # Load and format model metrics
        metrics = load_model_metrics()
        metrics_text = format_model_metrics(metrics, use_inr)
        
        # Format price in selected currency
        if use_inr:
            prediction_inr = prediction * EUR_TO_INR_RATE
            return f"₹{prediction_inr:.2f}", metrics_text
        else:
            return f"€{prediction:.2f}", metrics_text
    except Exception as e:
        return f"Error: {str(e)}", ""

# Create Gradio interface
with gr.Blocks(title="Hostel Price Predictor") as demo:
    gr.Markdown("# 🏨 Hostel Price Prediction")
    gr.Markdown("Predict the price per night for hostel beds based on various factors.")
    
    # Add currency toggle
    use_inr = gr.Checkbox(label="Display prices in Indian Rupees (₹)", value=True)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Location & Basic Information")
            city = gr.Dropdown(
                choices=[
                    "Berlin", "Paris", "London", "Amsterdam", "Barcelona",
                    "Rome", "Prague", "Vienna", "Budapest", "Dublin",
                    "Madrid", "Athens", "Lisbon", "Warsaw", "Stockholm",
                    "Copenhagen", "Oslo", "Helsinki", "Brussels", "Milan",
                    # Indian cities
                    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai",
                    "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Goa",
                    "Kochi", "Mysore", "Darjeeling", "Shimla", "Manali"
                ],
                value="Berlin",
                label="City"
            )
            distance_to_center = gr.Slider(0.0, 10.0, value=1.0, label="Distance to City Center (km)")
            rating = gr.Slider(1.0, 5.0, value=4.0, label="Hostel Rating")
            num_reviews = gr.Number(100, label="Number of Reviews")
            room_type = gr.Radio(["dorm", "private"], value="dorm", label="Room Type")
            beds_in_room = gr.Number(4, label="Beds in Room")
            room_area = gr.Number(150, label="Room Area (sq ft)")
            
        with gr.Column():
            gr.Markdown("### Amenities & Features")
            with gr.Row():
                with gr.Column():
                    breakfast = gr.Checkbox(True, label="Breakfast Included")
                    wifi = gr.Checkbox(True, label="WiFi Available")
                    laundry = gr.Checkbox(True, label="Laundry Service")
                    kitchen = gr.Checkbox(True, label="Kitchen Access")
                    ac = gr.Checkbox(True, label="Air Conditioning")
                
                with gr.Column():
                    locker = gr.Checkbox(True, label="Locker Available")
                    security = gr.Checkbox(True, label="24h Security")
                    female_dorm = gr.Checkbox(False, label="Female-only Dorm")
                    common_areas = gr.Number(2, label="Common Areas Count")
            
            gr.Markdown("### Booking Information")
            season = gr.Dropdown(["low", "shoulder", "peak"], value="peak", label="Travel Season")
            weekend = gr.Checkbox(False, label="Weekend Stay")
            occupancy_rate = gr.Slider(0.0, 1.0, value=0.7, label="Occupancy Rate")
            hostel_age = gr.Number(5, label="Hostel Age (years)")
            bathroom_type = gr.Dropdown(["shared", "private"], value="private", label="Bathroom Type")
    
    predict_btn = gr.Button("🔮 Predict Price")
    with gr.Row():
        output = gr.Textbox(label="Predicted Price per Night")
        metrics_output = gr.Markdown(label="Model Performance Metrics")
    
    predict_btn.click(
        predict_price,
        inputs=[
            city, distance_to_center, rating, num_reviews, room_type, beds_in_room,
            room_area, breakfast, wifi, laundry, kitchen, ac, locker, security,
            female_dorm, common_areas, season, weekend, occupancy_rate, hostel_age,
            bathroom_type, use_inr
        ],
        outputs=[output, metrics_output]
    )
    
    gr.Markdown("---")
    gr.Markdown("""
    ### About this App
    This application uses machine learning to predict hostel prices based on:
    - Location factors (city, distance to center)
    - Property features (age, amenities, security)
    - Room characteristics (type, beds, area)
    - Demand indicators (occupancy, reviews, season)
    - Engineered features (demand index, interaction terms)
    
    The model was trained on hostel data from major European and Indian cities and achieved high accuracy 
    in predicting prices across different scenarios.
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7863)
