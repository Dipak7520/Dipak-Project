# Hostel Price Prediction Project - Summary

## Project Overview

This project implements a machine learning solution to predict hostel bed prices based on various factors including location, amenities, demand indicators, and engineered features. The goal is to help hostel owners, travelers, and booking platforms estimate realistic prices.

## Key Components

### 1. Data Generation
- Created synthetic hostel data with realistic features
- Generated 2000 samples across 35 cities (20 European + 15 Indian)
- Included features like location, amenities, occupancy rates, and reviews
- Added price multiplier feature for configurable price adjustments

### 2. Feature Engineering
- Engineered advanced features including:
  - Demand Index (weighted combination of occupancy, reviews, and season)
  - Interaction features (rating × reviews, distance × season, etc.)
  - Density metrics (beds per room area)

### 3. Machine Learning Models
Trained and compared multiple regression models:
- Linear Regression
- Random Forest
- Gradient Boosting
- XGBoost
- CatBoost

### 4. Model Evaluation
Evaluated models using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

### 5. Model Interpretability
- Permutation feature importance analysis
- SHAP values for explainable AI
- Feature importance comparison across models

### 6. Web Applications
- Streamlit application for interactive price prediction
- Gradio application as an alternative interface
- Both applications display model performance metrics

## Project Structure

```
├── app/                    # Web applications
│   ├── predictor.py        # Streamlit app
│   └── gradio_app.py       # Gradio app
├── data/                   # Datasets
│   ├── hostel_data.csv     # Initial sample data
│   └── hostel_data_extended.csv  # Generated extended data
├── models/                 # Saved trained models
├── notebooks/              # Jupyter notebooks
│   └── eda_notebook.ipynb  # Exploratory data analysis
├── results/                # Generated visualizations
├── src/                    # Source code
│   ├── data_processing.py  # Data preprocessing
│   ├── data_generator.py   # Synthetic data generation
│   ├── model_training.py   # Model training and evaluation
│   ├── model_interpretability.py  # Model analysis
│   ├── visualization.py    # Data visualization
│   ├── save_encoders.py    # Utility functions
│   └── main.py            # Main execution script
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── run_analysis.py        # Simplified analysis script
└── run_project.py         # Project workflow script
```

## Key Insights

Based on the correlation analysis:
1. **Occupancy Rate** has the strongest positive correlation with price (0.29)
2. **Weekend stays** increase pricing (0.16 correlation)
3. **Ratings** have a moderate positive impact (0.09 correlation)
4. **Number of reviews** also positively correlates with price (0.08)

## How to Use This Project

### Installation
1. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Analysis
1. Generate extended dataset with price multiplier:
   ```bash
   python src/data_generator.py 1.2
   ```

2. Run simplified analysis:
   ```bash
   python run_analysis.py
   ```

### Running the Full Pipeline
1. Train models and generate results with price multiplier:
   ```bash
   python src/main.py 1.2
   ```

### Running Web Applications
1. Streamlit app:
   ```bash
   streamlit run app/predictor.py
   ```

2. Gradio app:
   ```bash
   python app/gradio_app.py
   ```

## Enhanced Features

### Indian Cities Support
The project now includes 15 major Indian cities with appropriate base prices:
- Mumbai, Delhi, Bangalore, Hyderabad, Chennai
- Kolkata, Pune, Ahmedabad, Jaipur, Goa
- Kochi, Mysore, Darjeeling, Shimla, Manali

### Price Multiplier
All generated prices can be increased by a configurable factor using the price multiplier parameter.

### Model Performance Metrics Display
Both web applications now display model performance metrics (MAE, RMSE, R² Score) directly on the interface.

## Future Improvements

1. Collect real-world hostel data for more accurate predictions
2. Implement advanced feature selection techniques
3. Add more sophisticated models like neural networks
4. Improve the web application UI/UX
5. Add more visualization options
6. Implement model monitoring and retraining pipelines

## Technologies Used

- Python
- Pandas, NumPy for data manipulation
- Scikit-learn for machine learning
- XGBoost and CatBoost for gradient boosting
- SHAP and LIME for model interpretability
- Streamlit and Gradio for web applications
- Matplotlib and Seaborn for visualization

This project demonstrates a complete machine learning workflow from data generation to deployment, with a focus on practical application in the hospitality industry.