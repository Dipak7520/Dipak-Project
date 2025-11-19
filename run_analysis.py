#!/usr/bin/env python3
"""
Simplified script to run the hostel price prediction analysis
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=== Hostel Price Prediction Analysis ===")
    
    # Load data
    data_path = os.path.join("data", "hostel_data_extended.csv")
    if not os.path.exists(data_path):
        # Try absolute path
        data_path = os.path.join(os.path.dirname(__file__), "data", "hostel_data_extended.csv")
        if not os.path.exists(data_path):
            print("Data file not found. Please run data generation first.")
            return
    
    df = pd.read_csv(data_path)
    print(f"Loaded data with {len(df)} samples")
    print(f"Price range: €{df['price_per_night'].min():.2f} - €{df['price_per_night'].max():.2f}")
    
    # Show city distribution
    print(f"\nCities represented: {df['city'].nunique()}")
    print("Top 10 cities by sample count:")
    print(df['city'].value_counts().head(10))
    
    # Show basic statistics
    print("\n=== Basic Statistics ===")
    print(df.describe())
    
    # Show correlation with price
    print("\n=== Correlation with Price ===")
    correlations = df.select_dtypes(include=[np.number]).corr()['price_per_night'].sort_values(ascending=False)
    print(correlations.head(10))
    
    # Show average prices by room type
    print("\n=== Average Price by Room Type ===")
    print(df.groupby('room_type')['price_per_night'].agg(['count', 'mean', 'median']))
    
    # Show average prices by season
    print("\n=== Average Price by Season ===")
    print(df.groupby('season')['price_per_night'].agg(['count', 'mean', 'median']))
    
    print("\n=== Analysis Complete ===")
    print("For full model training, please run 'python src/main.py' from the src directory")

if __name__ == "__main__":
    main()