# Hostel Price Prediction Project

This project predicts hostel bed prices using machine learning techniques. It analyzes various factors like location, amenities, demand, and reviews to predict pricing.

## Project Overview

The goal of this project is to build an intelligent machine learning model that predicts the nightly bed price of hostels using a combination of location, room features, amenities, demand factors, reviews, and advanced engineered attributes. The project also includes model interpretability analysis and a web application for real-time price prediction.

## Project Structure

```
├── app/                    # Web applications (Streamlit and Gradio)
├── data/                   # Datasets
├── models/                 # Saved trained models and encoders
├── notebooks/              # Jupyter notebooks for analysis
├── results/                # Generated visualizations and results
├── src/                    # Source code
│   ├── data_processing.py  # Data preprocessing and feature engineering
│   ├── data_generator.py   # Synthetic data generation
│   ├── model_training.py   # Model training and evaluation
│   ├── model_interpretability.py  # Model interpretability analysis
│   ├── save_encoders.py    # Utility to save encoders
│   ├── visualization.py    # Data visualization functions
│   └── main.py            # Main execution script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Features

### Dataset Features

#### Basic Features
- City (European and Indian cities)
- Distance to center (km)
- Rating
- Number of reviews
- Room type (dorm/private)
- Beds in room
- Breakfast included
- WiFi availability
- Laundry service
- Season (low/shoulder/peak)
- Weekend stay
- Occupancy rate
- Price per night (Target variable)

#### Hostel Property Features
- Hostel age (years)
- Kitchen access
- Air conditioning
- Locker availability
- 24h security
- Female-only dorm availability
- Common area count
- Room area (sq ft)
- Bathroom type

### Engineered Features
- Rating × Number of reviews (confidence-weighted rating)
- Beds in room / Room area = Density
- Distance to center × Season
- Rating × Occupancy rate
- Demand Index = 0.4 × Occupancy rate + 0.3 × Recent reviews + 0.3 × Season factor

## Models Implemented

1. Linear Regression
2. Random Forest
3. Gradient Boosting
4. XGBoost
5. CatBoost

## Evaluation Metrics

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

## Requirements

- Python 3.7+
- All dependencies listed in [requirements.txt](requirements.txt)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd hostel-price-prediction
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Generate Extended Dataset (Optional)
```bash
python src/data_generator.py [price_multiplier]
```

### 2. Run the Main Analysis
```bash
python src/main.py [price_multiplier]
```

This will:
- Generate data with optional price multiplier
- Preprocess the data
- Train all models
- Compare model performance
- Perform interpretability analysis
- Save results and visualizations

### 3. Run Jupyter Notebook for EDA
```bash
jupyter notebook notebooks/eda_notebook.ipynb
```

### 4. Run Streamlit Web Application
```bash
streamlit run app/predictor.py
```

### 5. Run Gradio Web Application
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
All generated prices can be increased by a configurable factor using the `--price_multiplier` option.

### Model Performance Metrics Display
Both web applications now display model performance metrics (MAE, RMSE, R² Score) directly on the interface.

## Results

The models are evaluated and compared using MAE, RMSE, and R² Score. Feature importance is analyzed using permutation importance and SHAP values.

## Model Interpretability

The project uses:
- Permutation Feature Importance
- SHAP (SHapley Additive exPlanations) values
- Feature importance comparison across models

## Web Application

The project includes two web applications for real-time price prediction:
1. Streamlit application (`app/predictor.py`)
2. Gradio application (`app/gradio_app.py`)

Both applications allow users to input hostel and booking details to get price predictions and display model performance metrics.

## Project Workflow

1. **Data Collection**: Synthetic hostel data is generated for European and Indian cities
2. **Data Preprocessing**: Cleaning, encoding, and feature engineering
3. **Feature Engineering**: Creating advanced features like demand index and interaction terms
4. **Model Training**: Training multiple ML models
5. **Model Evaluation**: Comparing models using various metrics
6. **Model Interpretability**: Understanding feature importance
7. **Deployment**: Web application for real-time predictions

## Key Insights

- Location factors (city, distance to center) significantly impact pricing
- Amenities like breakfast, air conditioning, and 24h security increase prices
- Demand factors (occupancy rate, season) are crucial for pricing
- Engineered features like demand index improve model performance
- Room type (dorm vs private) has a major impact on pricing

## Future Improvements

- Collect real-world hostel data for more accurate predictions
- Implement advanced feature selection techniques
- Add more sophisticated models like neural networks
- Improve the web application UI/UX
- Add more visualization options
- Implement model monitoring and retraining pipelines