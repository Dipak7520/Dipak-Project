import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

def calculate_shap_values(model, X_train, X_test, model_name=""):
    """
    Calculate SHAP values for a trained model
    
    Args:
        model: Trained model
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        model_name (str): Name of the model for labeling
        
    Returns:
        shap.Explanation: SHAP explanation object
    """
    try:
        # Create SHAP explainer based on model type
        if model_name.lower() in ['linear regression']:
            explainer = shap.LinearExplainer(model, X_train)
        elif model_name.lower() in ['random forest', 'gradient boosting', 'xgboost', 'catboost']:
            explainer = shap.TreeExplainer(model)
        else:
            # For other models, use permutation explainer
            explainer = shap.PermutationExplainer(model.predict, X_train)
        
        # Calculate SHAP values for test set
        shap_values = explainer.shap_values(X_test)
        
        return shap_values, explainer
    except Exception as e:
        print(f"Error calculating SHAP values for {model_name}: {str(e)}")
        return None, None

def plot_feature_importance_shap(shap_values, feature_names, model_name="", top_n=15):
    """
    Plot SHAP feature importance
    
    Args:
        shap_values: SHAP values
        feature_names (list): List of feature names
        model_name (str): Name of the model for labeling
        top_n (int): Number of top features to display
    """
    try:
        # For tree models, shap_values might be a list
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        # Create SHAP summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, feature_names=feature_names, show=False, max_display=top_n)
        plt.title(f'SHAP Feature Importance - {model_name}')
        plt.tight_layout()
        plt.savefig(f'../results/shap_summary_{model_name.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"Error plotting SHAP values for {model_name}: {str(e)}")

def plot_feature_importance_permutation(model, X_test, y_test, feature_names, model_name="", n_repeats=10):
    """
    Calculate and plot permutation feature importance
    
    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        feature_names (list): List of feature names
        model_name (str): Name of the model for labeling
        n_repeats (int): Number of repeats for permutation
    """
    try:
        # Calculate permutation importance
        perm_importance = permutation_importance(model, X_test, y_test, n_repeats=n_repeats, random_state=42)
        
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': perm_importance.importances_mean,
            'std': perm_importance.importances_std
        }).sort_values('importance', ascending=False).head(15)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
        plt.xlabel('Permutation Importance')
        plt.title(f'Permutation Feature Importance - {model_name}')
        plt.tight_layout()
        plt.savefig(f'../results/permutation_importance_{model_name.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return importance_df
    except Exception as e:
        print(f"Error calculating permutation importance for {model_name}: {str(e)}")
        return None

def analyze_model_interpretability(model, model_name, X_train, X_test, y_test, feature_names):
    """
    Comprehensive interpretability analysis for a model
    
    Args:
        model: Trained model
        model_name (str): Name of the model
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        feature_names (list): List of feature names
    """
    print(f"\n=== Interpretability Analysis for {model_name} ===")
    
    # Create results directory if it doesn't exist
    import os
    if not os.path.exists('../results'):
        os.makedirs('../results')
    
    # 1. Permutation Feature Importance
    print("Calculating permutation feature importance...")
    perm_importance_df = plot_feature_importance_permutation(
        model, X_test, y_test, feature_names, model_name
    )
    
    # 2. SHAP Values (if applicable)
    print("Calculating SHAP values...")
    shap_values, explainer = calculate_shap_values(model, X_train, X_test, model_name)
    
    if shap_values is not None:
        print("Plotting SHAP feature importance...")
        plot_feature_importance_shap(shap_values, feature_names, model_name)
    
    return perm_importance_df, shap_values

def compare_feature_importance(all_models, X_train, X_test, y_test, feature_names):
    """
    Compare feature importance across all models
    
    Args:
        all_models (dict): Dictionary of trained models
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        feature_names (list): List of feature names
    """
    # Create results directory if it doesn't exist
    import os
    if not os.path.exists('../results'):
        os.makedirs('../results')
    
    # Collect feature importances from all models
    importance_data = []
    
    for model_name, model in all_models.items():
        try:
            # Calculate permutation importance
            perm_importance = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42)
            
            # Get top 10 features for this model
            top_indices = np.argsort(perm_importance.importances_mean)[::-1][:10]
            
            for i, idx in enumerate(top_indices):
                importance_data.append({
                    'Model': model_name,
                    'Feature': feature_names[idx],
                    'Importance': perm_importance.importances_mean[idx],
                    'Rank': i + 1
                })
        except Exception as e:
            print(f"Error with {model_name}: {str(e)}")
    
    # Create DataFrame and pivot for heatmap
    if importance_data:
        importance_df = pd.DataFrame(importance_data)
        
        # Create heatmap of feature importance across models
        pivot_df = importance_df.pivot_table(
            index='Feature', 
            columns='Model', 
            values='Importance', 
            fill_value=0
        )
        
        # Plot heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(pivot_df, annot=True, cmap='YlOrRd', fmt='.3f')
        plt.title('Feature Importance Comparison Across Models')
        plt.tight_layout()
        plt.savefig('../results/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return importance_df
    else:
        print("No feature importance data to display")
        return None

if __name__ == "__main__":
    print("Model interpretability module ready!")
    print("Use analyze_model_interpretability() for individual model analysis")
    print("Use compare_feature_importance() to compare models")