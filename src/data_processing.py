import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

# Currency conversion rate (EUR to INR)
# As of November 2025, 1 EUR ≈ 90 INR (approximate rate)
EUR_TO_INR_RATE = 90.0

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the hostel data
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Processed dataframe
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Handle missing values if any
    data = data.fillna(method='ffill')
    
    return data

def encode_categorical_features(data):
    """
    Encode categorical features using label encoding
    
    Args:
        data (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with encoded features
    """
    # Create a copy
    df = data.copy()
    
    # Label encode categorical variables
    le_city = LabelEncoder()
    le_room_type = LabelEncoder()
    le_season = LabelEncoder()
    le_bathroom = LabelEncoder()
    
    df['city_encoded'] = le_city.fit_transform(df['city'])
    df['room_type_encoded'] = le_room_type.fit_transform(df['room_type'])
    df['season_encoded'] = le_season.fit_transform(df['season'])
    df['bathroom_type_encoded'] = le_bathroom.fit_transform(df['bathroom_type'])
    
    # Store encoders for later use in predictions
    encoders = {
        'city': le_city,
        'room_type': le_room_type,
        'season': le_season,
        'bathroom_type': le_bathroom
    }
    
    return df, encoders

def engineer_features(data):
    """
    Engineer advanced features for the model
    
    Args:
        data (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with engineered features
    """
    # Create a copy
    df = data.copy()
    
    # Convert season to numeric factor
    season_mapping = {'low': 1, 'shoulder': 2, 'peak': 3}
    df['season_factor'] = df['season'].map(season_mapping)
    
    # Create interaction features
    df['rating_num_reviews'] = df['rating'] * df['num_reviews']
    df['density'] = df['beds_in_room'] / df['room_area_sqft']
    df['distance_season'] = df['distance_to_center_km'] * df['season_factor']
    df['rating_occupancy'] = df['rating'] * df['occupancy_rate']
    
    # Create demand index
    # Normalize num_reviews to create num_recent_reviews proxy
    df['num_recent_reviews'] = (df['num_reviews'] - df['num_reviews'].min()) / (df['num_reviews'].max() - df['num_reviews'].min())
    
    # Calculate demand index
    df['demand_index'] = (
        0.4 * df['occupancy_rate'] +
        0.3 * df['num_recent_reviews'] +
        0.3 * df['season_factor'] / 3  # Normalize season factor
    )
    
    # Clamp negative values to zero for all numeric features
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        df[col] = df[col].clip(lower=0)
    
    return df

def prepare_features_and_target(data):
    """
    Prepare features and target variables for modeling
    
    Args:
        data (pd.DataFrame): Input dataframe with engineered features
        
    Returns:
        tuple: (X, y) features and target
    """
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
    
    X = data[feature_columns]
    y = data['price_per_night']
    
    # Clamp negative values to zero
    X = X.clip(lower=0)
    
    return X, y

def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        
    Returns:
        tuple: Scaled training and test features, scaler object
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to preserve column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Clamp negative values to zero after scaling
    X_train_scaled = X_train_scaled.clip(lower=0)
    X_test_scaled = X_test_scaled.clip(lower=0)
    
    return X_train_scaled, X_test_scaled, scaler

def convert_price_to_inr(price_eur, rate=EUR_TO_INR_RATE):
    """
    Convert price from EUR to INR
    
    Args:
        price_eur (float): Price in EUR
        rate (float): Conversion rate from EUR to INR
        
    Returns:
        float: Price in INR
    """
    return price_eur * rate

def prepare_data_for_modeling(file_path, test_size=0.2, random_state=42):
    """
    Complete pipeline to prepare data for modeling
    
    Args:
        file_path (str): Path to the CSV file
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scalers, encoders)
    """
    # Load and preprocess data
    data = load_and_preprocess_data(file_path)
    
    # Encode categorical features
    data, encoders = encode_categorical_features(data)
    
    # Engineer features
    data = engineer_features(data)
    
    # Prepare features and target
    X, y = prepare_features_and_target(data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Save encoders and scaler for later use
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    import joblib
    for name, encoder in encoders.items():
        joblib.dump(encoder, os.path.join(models_dir, f"{name}_encoder.pkl"))
    
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, encoders, data

if __name__ == "__main__":
    # Example usage
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "data", "hostel_data_extended.csv")
    X_train, X_test, y_train, y_test, scaler, encoders, full_data = prepare_data_for_modeling(
        data_path
    )
    
    print("Data preprocessing complete!")
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    print(f"Features: {list(X_train.columns)}")
    
    # Example of price conversion
    sample_price_eur = 25.50
    sample_price_inr = convert_price_to_inr(sample_price_eur)
    print(f"Sample price conversion: €{sample_price_eur:.2f} = ₹{sample_price_inr:.2f}")