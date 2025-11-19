import pandas as pd
import numpy as np
import os
import sys
import warnings
import json
warnings.filterwarnings('ignore')

# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from src.data_processing import prepare_data_for_modeling
from src.model_training import train_all_models
from src.model_interpretability import analyze_model_interpretability, compare_feature_importance

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

def main(price_multiplier=1.0):
    """
    Main function to run the hostel price prediction project
    
    Args:
        price_multiplier (float): Factor to multiply all prices by
    """
    print("=== Hostel Price Prediction Project ===")
    print("Starting data preprocessing...")
    
    # Generate data with price multiplier
    print(f"Generating data with price multiplier: {price_multiplier}")
    from src.data_generator import generate_hostel_data
    
    # Generate extended dataset with price multiplier
    df = generate_hostel_data(2000, price_multiplier=price_multiplier)
    
    # Save generated data
    data_path = os.path.join(project_root, "data", "hostel_data_extended.csv")
    df.to_csv(data_path, index=False)
    print(f"Data saved to {data_path}")
    
    # Prepare data for modeling
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
    
    # Select best model for interpretability analysis
    best_model_name = results['comparison_results'].iloc[0]['Model']
    best_model = results['models'][best_model_name]
    
    print(f"\n=== Analyzing Best Model ({best_model_name}) ===")
    
    # Analyze interpretability of the best model
    feature_names = list(X_train.columns)
    perm_importance, shap_values = analyze_model_interpretability(
        best_model, best_model_name, X_train, X_test, y_test, feature_names
    )
    
    # Compare feature importance across all models
    print("\n=== Comparing Feature Importance Across Models ===")
    importance_comparison = compare_feature_importance(
        results['models'], X_train, X_test, y_test, feature_names
    )
    
    # Save results to CSV
    results_dir = os.path.join(project_root, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    results['comparison_results'].to_csv(os.path.join(results_dir, "model_comparison.csv"), index=False)
    print("\nResults saved to ../results/model_comparison.csv")
    
    if importance_comparison is not None:
        importance_comparison.to_csv(os.path.join(results_dir, "feature_importance_comparison.csv"), index=False)
        print("Feature importance comparison saved to ../results/feature_importance_comparison.csv")
    
    print("\n=== Project Complete ===")
    print("Check the results folder for model performance metrics and visualizations.")
    print("Trained models are saved in the models folder.")

if __name__ == "__main__":
    # Check for price multiplier argument
    price_multiplier = 1.0
    if len(sys.argv) > 1:
        try:
            price_multiplier = float(sys.argv[1])
        except ValueError:
            print("Invalid price multiplier. Using default value of 1.0")
    
    main(price_multiplier)