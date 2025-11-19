import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def plot_model_comparison(results_df):
    """
    Plot model comparison results
    
    Args:
        results_df (pd.DataFrame): DataFrame with model comparison results
    """
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot MAE
    axes[0].bar(results_df['Model'], results_df['MAE'], color='skyblue')
    axes[0].set_title('Mean Absolute Error (MAE)')
    axes[0].set_ylabel('MAE')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Plot RMSE
    axes[1].bar(results_df['Model'], results_df['RMSE'], color='lightcoral')
    axes[1].set_title('Root Mean Squared Error (RMSE)')
    axes[1].set_ylabel('RMSE')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Plot R2
    axes[2].bar(results_df['Model'], results_df['R2'], color='lightgreen')
    axes[2].set_title('R² Score')
    axes[2].set_ylabel('R²')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('../results/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_actual_vs_predicted(y_test, y_pred, model_name=""):
    """
    Plot actual vs predicted values
    
    Args:
        y_test (array): Actual values
        y_pred (array): Predicted values
        model_name (str): Name of the model for labeling
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price (€)')
    plt.ylabel('Predicted Price (€)')
    plt.title(f'Actual vs Predicted Prices - {model_name}')
    plt.grid(True, alpha=0.3)
    
    # Add metrics as text
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    plt.text(0.05, 0.95, f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.2f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'../results/actual_vs_predicted_{model_name.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_importance(feature_names, importance_scores, top_n=15):
    """
    Plot feature importance
    
    Args:
        feature_names (list): List of feature names
        importance_scores (array): Importance scores
        top_n (int): Number of top features to display
    """
    # Create DataFrame and sort by importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    }).sort_values('importance', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
    plt.xlabel('Importance Score')
    plt.title('Top Feature Importances')
    plt.tight_layout()
    plt.savefig('../results/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_price_distribution(df):
    """
    Plot the distribution of prices
    
    Args:
        df (pd.DataFrame): DataFrame with price data
    """
    plt.figure(figsize=(10, 6))
    plt.hist(df['price_per_night'], bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Price per Night (€)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Hostel Bed Prices')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../results/price_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_price_by_category(df, category_col, title_suffix=""):
    """
    Plot price distribution by category
    
    Args:
        df (pd.DataFrame): DataFrame with data
        category_col (str): Column to group by
        title_suffix (str): Suffix for plot title
    """
    plt.figure(figsize=(12, 6))
    df.boxplot(column='price_per_night', by=category_col, ax=plt.gca())
    plt.title(f'Price Distribution by {title_suffix}')
    plt.suptitle('')  # Remove default title
    plt.xlabel(category_col.replace('_', ' ').title())
    plt.ylabel('Price per Night (€)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'../results/price_by_{category_col}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_correlation_heatmap(df, figsize=(15, 12)):
    """
    Plot correlation heatmap of numerical features
    
    Args:
        df (pd.DataFrame): DataFrame with numerical features
        figsize (tuple): Figure size
    """
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Calculate correlation matrix
    corr_matrix = df[numerical_cols].corr()
    
    # Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', 
                cbar_kws={'shrink': 0.8})
    plt.title('Correlation Matrix of Hostel Features')
    plt.tight_layout()
    plt.savefig('../results/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_comprehensive_report(df, results_df):
    """
    Create a comprehensive visualization report
    
    Args:
        df (pd.DataFrame): Original data
        results_df (pd.DataFrame): Model comparison results
    """
    # Create results directory if it doesn't exist
    import os
    if not os.path.exists('../results'):
        os.makedirs('../results')
    
    print("Generating comprehensive visualization report...")
    
    # 1. Model comparison
    plot_model_comparison(results_df)
    
    # 2. Price distribution
    plot_price_distribution(df)
    
    # 3. Price by room type
    plot_price_by_category(df, 'room_type', 'Room Type')
    
    # 4. Price by season
    plot_price_by_category(df, 'season', 'Season')
    
    # 5. Correlation heatmap
    plot_correlation_heatmap(df)
    
    print("Visualization report complete. Check the 'results' folder for generated plots.")

if __name__ == "__main__":
    print("Visualization module ready!")
    print("Use create_comprehensive_report() to generate all visualizations.")