import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processing import convert_price_to_inr, engineer_features

def test_currency_conversion():
    """Test the currency conversion function"""
    print("Testing currency conversion...")
    
    # Test conversion from EUR to INR
    eur_price = 25.50
    inr_price = convert_price_to_inr(eur_price)
    
    print(f"€{eur_price:.2f} = ₹{inr_price:.2f}")
    
    # Check if conversion is correct (using rate of 90)
    expected_inr = eur_price * 90.0
    assert abs(inr_price - expected_inr) < 0.01, f"Conversion error: expected {expected_inr}, got {inr_price}"
    
    print("✓ Currency conversion test passed!")

def test_negative_value_clamping():
    """Test that negative values are clamped to zero"""
    print("\nTesting negative value clamping...")
    
    # Create a test dataframe with some negative values
    # Include all required columns for feature engineering
    test_data = pd.DataFrame({
        'city': ['Berlin', 'Paris', 'London'],
        'distance_to_center_km': [1.5, -0.5, 3.0],
        'rating': [4.2, 5.0, -1.0],
        'num_reviews': [100, 200, -50],
        'room_type': ['dorm', 'private', 'dorm'],
        'beds_in_room': [4, 6, -2],
        'breakfast_included': [1, 0, 1],
        'wifi': [1, 1, 0],
        'laundry': [1, 0, 1],
        'weekend': [0, 1, 0],
        'occupancy_rate': [0.7, 0.9, -0.2],
        'hostel_age_years': [5, 10, -2],
        'kitchen_access': [1, 1, 0],
        'air_conditioning': [1, 0, 1],
        'locker_available': [1, 1, 0],
        'security_24h': [1, 1, 1],
        'female_only_dorm_available': [0, 1, 0],
        'common_area_count': [2, 3, -1],
        'room_area_sqft': [150, 200, 100],
        'season': ['peak', 'low', 'shoulder'],
        'bathroom_type': ['private', 'shared', 'private'],
        'price_per_night': [25.5, 30.0, -5.0]
    })
    
    print("Original data:")
    print(test_data[['distance_to_center_km', 'rating', 'num_reviews', 'beds_in_room', 'occupancy_rate', 'price_per_night']])
    
    # Apply feature engineering
    processed_data = engineer_features(test_data)
    
    print("\nAfter feature engineering and clamping (showing key columns):")
    print(processed_data[['distance_to_center_km', 'rating', 'num_reviews', 'beds_in_room', 'occupancy_rate', 'price_per_night', 'demand_index']].head())
    
    # Check that all values are non-negative
    numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        assert (processed_data[col] >= 0).all(), f"Found negative values in column {col}"
    
    print("✓ Negative value clamping test passed!")

def test_streamlit_preprocessing():
    """Test the Streamlit preprocessing function"""
    print("\nTesting Streamlit preprocessing...")
    
    # Import the Streamlit preprocessing function
    sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))
    
    # Create mock encoders and scaler
    class MockEncoder:
        def transform(self, x):
            return [0] * len(x)
    
    class MockScaler:
        def transform(self, x):
            return x
    
    encoders = {
        'city': MockEncoder(),
        'room_type': MockEncoder(),
        'season': MockEncoder(),
        'bathroom_type': MockEncoder()
    }
    
    scaler = MockScaler()
    
    # Import the actual preprocessing function
    from app.predictor import preprocess_input
    
    # Test data with potential negative values
    user_inputs = {
        'city': 'Berlin',
        'distance_to_center_km': -1.0,  # Negative value
        'rating': 4.0,
        'num_reviews': 100,
        'room_type': 'dorm',
        'beds_in_room': 4,
        'room_area_sqft': 150,
        'breakfast_included': 1,
        'wifi': 1,
        'laundry': 1,
        'kitchen_access': 1,
        'air_conditioning': 1,
        'locker_available': 1,
        'security_24h': 1,
        'female_only_dorm_available': 0,
        'common_area_count': 2,
        'season': 'peak',
        'weekend': 0,
        'occupancy_rate': 0.7,
        'hostel_age_years': 5,
        'bathroom_type': 'private'
    }
    
    # Preprocess the inputs
    processed_features = preprocess_input(user_inputs, encoders, scaler)
    
    print("Processed features (first 5 columns):")
    print(processed_features.iloc[:, :5])
    
    # Check that all values are non-negative
    assert (processed_features >= 0).all().all(), "Found negative values in processed features"
    
    print("✓ Streamlit preprocessing test passed!")

if __name__ == "__main__":
    print("Running tests for currency conversion and negative value clamping...\n")
    
    try:
        test_currency_conversion()
        test_negative_value_clamping()
        test_streamlit_preprocessing()
        
        print("\n🎉 All tests passed! The fixes are working correctly.")
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        raise