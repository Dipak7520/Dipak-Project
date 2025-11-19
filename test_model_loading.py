import joblib
import os
import json

def test_model_loading():
    """Test that all model files can be loaded correctly"""
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    print("Testing model loading...")
    print(f"Project root: {project_root}")
    
    # Test main model
    model_path = os.path.join(project_root, "models", "random_forest.pkl")
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            print(f"✓ Successfully loaded random_forest.pkl")
            print(f"  Model type: {type(model)}")
        except Exception as e:
            print(f"✗ Error loading random_forest.pkl: {e}")
    else:
        print(f"✗ Model file not found: {model_path}")
    
    # Test encoders
    encoders = ["city_encoder.pkl", "room_type_encoder.pkl", "season_encoder.pkl", "bathroom_type_encoder.pkl"]
    for encoder_name in encoders:
        encoder_path = os.path.join(project_root, "models", encoder_name)
        if os.path.exists(encoder_path):
            try:
                encoder = joblib.load(encoder_path)
                print(f"✓ Successfully loaded {encoder_name}")
            except Exception as e:
                print(f"✗ Error loading {encoder_name}: {e}")
        else:
            print(f"✗ Encoder file not found: {encoder_path}")
    
    # Test scaler
    scaler_path = os.path.join(project_root, "models", "scaler.pkl")
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
            print(f"✓ Successfully loaded scaler.pkl")
        except Exception as e:
            print(f"✗ Error loading scaler.pkl: {e}")
    else:
        print(f"✗ Scaler file not found: {scaler_path}")
    
    # Test metrics
    metrics_path = os.path.join(project_root, "models", "model_metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            print(f"✓ Successfully loaded model_metrics.json")
            print(f"  Models in metrics: {list(metrics.keys())}")
        except Exception as e:
            print(f"✗ Error loading model_metrics.json: {e}")
    else:
        print(f"✗ Metrics file not found: {metrics_path}")

if __name__ == "__main__":
    test_model_loading()