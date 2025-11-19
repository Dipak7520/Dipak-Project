import pandas as pd
import numpy as np
import os
import sys
import warnings
import json
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.data_processing import prepare_data_for_modeling
from src.model_training import train_all_models

def save_model_metrics(metrics_df, filepath):
    """
    Save model metrics to a JSON file for use in web applications
    
    Args:
        metrics_df (pd.DataFrame): DataFrame with model comparison results
        filepath (str): Path to save the metrics
    """
    # Convert DataFrame to dictionary
    metrics_dict = {}
    for _, row in metrics_df.iterrows():
        model_name = row['Model']
        metrics_dict[model_name] = {
            'MAE': float(row['MAE']),
            'RMSE': float(row['RMSE']),
            'R2': float(row['R2'])
        }
    
    # Save to JSON file
    with open(filepath, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    print(f"Model metrics saved to {filepath}")

def main():
    print("=== Hostel Price Prediction Project ===")
    print("Training models...")
    
    # Get project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Prepare data for modeling
    data_path = os.path.join(project_root, "data", "hostel_data_extended.csv")
    X_train, X_test, y_train, y_test, scaler, encoders, full_data = prepare_data_for_modeling(
        data_path
    )
    
    print(f"Data preprocessing complete!")
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Train all models
    print("\nTraining and comparing models...")
    results = train_all_models(X_train, y_train, X_test, y_test, save_models=True)
    
    # Display comparison results
    print("\n=== Model Performance Comparison ===")
    print(results['comparison_results'].to_string(index=False))
    
    # Save model metrics for web applications
    metrics_path = os.path.join(project_root, "models", "model_metrics.json")
    save_model_metrics(results['comparison_results'], metrics_path)
    
    print("\n=== Project Complete ===")
    print("Trained models are saved in the models folder.")
    print("Model metrics saved for web applications.")

if __name__ == "__main__":
    main()