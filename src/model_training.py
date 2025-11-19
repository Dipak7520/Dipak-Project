import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import joblib
import os

def train_linear_regression(X_train, y_train):
    """
    Train a Linear Regression model
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        
    Returns:
        LinearRegression: Trained model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, random_state=42):
    """
    Train a Random Forest model
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        random_state (int): Random seed for reproducibility
        
    Returns:
        RandomForestRegressor: Trained model
    """
    model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def train_gradient_boosting(X_train, y_train, random_state=42):
    """
    Train a Gradient Boosting model
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        random_state (int): Random seed for reproducibility
        
    Returns:
        GradientBoostingRegressor: Trained model
    """
    model = GradientBoostingRegressor(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train, random_state=42):
    """
    Train an XGBoost model
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        random_state (int): Random seed for reproducibility
        
    Returns:
        XGBRegressor: Trained model
    """
    model = XGBRegressor(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def train_catboost(X_train, y_train, random_state=42):
    """
    Train a CatBoost model
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        random_state (int): Random seed for reproducibility
        
    Returns:
        CatBoostRegressor: Trained model
    """
    model = CatBoostRegressor(iterations=100, random_seed=random_state, verbose=False)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model using multiple metrics
    
    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }

def compare_models(models_dict, X_test, y_test):
    """
    Compare multiple models and return their performance metrics
    
    Args:
        models_dict (dict): Dictionary of model names and trained models
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        
    Returns:
        pd.DataFrame: DataFrame with model comparison results
    """
    results = []
    
    for name, model in models_dict.items():
        metrics = evaluate_model(model, X_test, y_test)
        metrics['Model'] = name
        results.append(metrics)
    
    # Convert to DataFrame and sort by R2 score
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('R2', ascending=False)
    
    return results_df

def save_model(model, model_name, save_dir='../models'):
    """
    Save a trained model to disk
    
    Args:
        model: Trained model
        model_name (str): Name of the model
        save_dir (str): Directory to save the model
    """
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save model
    model_path = os.path.join(save_dir, f'{model_name}.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def load_model(model_name, save_dir='../models'):
    """
    Load a trained model from disk
    
    Args:
        model_name (str): Name of the model
        save_dir (str): Directory where the model is saved
        
    Returns:
        Loaded model
    """
    model_path = os.path.join(save_dir, f'{model_name}.pkl')
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model

def train_all_models(X_train, y_train, X_test, y_test, save_models=True):
    """
    Train all models and compare their performance
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        save_models (bool): Whether to save trained models
        
    Returns:
        dict: Dictionary containing trained models and comparison results
    """
    # Train models
    print("Training Linear Regression...")
    lr_model = train_linear_regression(X_train, y_train)
    
    print("Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    
    print("Training Gradient Boosting...")
    gb_model = train_gradient_boosting(X_train, y_train)
    
    print("Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)
    
    print("Training CatBoost...")
    cb_model = train_catboost(X_train, y_train)
    
    # Store models in dictionary
    models = {
        'Linear Regression': lr_model,
        'Random Forest': rf_model,
        'Gradient Boosting': gb_model,
        'XGBoost': xgb_model,
        'CatBoost': cb_model
    }
    
    # Compare models
    print("Comparing models...")
    comparison_results = compare_models(models, X_test, y_test)
    
    # Save models if requested
    if save_models:
        print("Saving models...")
        for name, model in models.items():
            # Clean model name for file saving
            clean_name = name.replace(' ', '_').lower()
            save_model(model, clean_name)
    
    return {
        'models': models,
        'comparison_results': comparison_results
    }

if __name__ == "__main__":
    # This would be run from the main script after data preparation
    print("Model training module ready!")
    print("Use train_all_models() function to train and compare all models.")