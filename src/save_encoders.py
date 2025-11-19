import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

def create_and_save_encoders(data_file='../data/hostel_data.csv', save_dir='../models'):
    """
    Create and save label encoders for categorical variables
    
    Args:
        data_file (str): Path to the data file
        save_dir (str): Directory to save encoders
    """
    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Load data
    df = pd.read_csv(data_file)
    
    # Create label encoders for categorical variables
    categorical_columns = ['city', 'room_type', 'season', 'bathroom_type']
    encoders = {}
    
    for col in categorical_columns:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col])
        encoders[col] = le
        
        # Save encoder
        encoder_path = os.path.join(save_dir, f'{col}_encoder.pkl')
        joblib.dump(le, encoder_path)
        print(f"Saved {col} encoder to {encoder_path}")
    
    # Create and save a scaler
    feature_columns = [
        'distance_to_center_km', 'rating', 'num_reviews', 'beds_in_room',
        'breakfast_included', 'wifi', 'laundry', 'weekend', 'occupancy_rate',
        'hostel_age_years', 'kitchen_access', 'air_conditioning', 
        'locker_available', 'security_24h', 'female_only_dorm_available',
        'common_area_count', 'room_area_sqft', 'city_encoded', 
        'room_type_encoded', 'season_encoded', 'bathroom_type_encoded'
    ]
    
    # Add engineered features that we'll create during preprocessing
    feature_columns.extend([
        'rating_num_reviews', 'density', 'distance_season', 
        'rating_occupancy', 'demand_index'
    ])
    
    # Create dummy data for fitting the scaler
    scaler = StandardScaler()
    
    # Save scaler
    scaler_path = os.path.join(save_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to {scaler_path}")
    
    print("All encoders and scaler saved successfully!")

def load_and_test_encoders():
    """
    Load and test the saved encoders
    """
    try:
        # Load encoders
        city_encoder = joblib.load('../models/city_encoder.pkl')
        room_type_encoder = joblib.load('../models/room_type_encoder.pkl')
        season_encoder = joblib.load('../models/season_encoder.pkl')
        bathroom_encoder = joblib.load('../models/bathroom_type_encoder.pkl')
        
        # Test encoding
        test_city = "Paris"
        test_room_type = "dorm"
        test_season = "peak"
        test_bathroom = "private"
        
        city_encoded = city_encoder.transform([test_city])[0]
        room_type_encoded = room_type_encoder.transform([test_room_type])[0]
        season_encoded = season_encoder.transform([test_season])[0]
        bathroom_encoded = bathroom_encoder.transform([test_bathroom])[0]
        
        print(f"\nEncoder Test Results:")
        print(f"City '{test_city}' -> {city_encoded}")
        print(f"Room Type '{test_room_type}' -> {room_type_encoded}")
        print(f"Season '{test_season}' -> {season_encoded}")
        print(f"Bathroom '{test_bathroom}' -> {bathroom_encoded}")
        
        return True
    except Exception as e:
        print(f"Error loading or testing encoders: {str(e)}")
        return False

if __name__ == "__main__":
    print("Creating and saving encoders...")
    create_and_save_encoders()
    
    print("\nTesting loaded encoders...")
    load_and_test_encoders()