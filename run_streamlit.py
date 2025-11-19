#!/usr/bin/env python3
"""
Script to run the Streamlit web application for hostel price prediction
"""

import subprocess
import sys
import os

def main():
    print("🏨 Hostel Price Prediction - Streamlit App")
    print("=" * 50)
    
    # Check if Streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("Streamlit is not installed. Please install it with:")
        print("pip install streamlit")
        sys.exit(1)
    
    # Get the path to the Streamlit app
    app_path = os.path.join("app", "predictor.py")
    
    if not os.path.exists(app_path):
        print(f"Streamlit app not found at {app_path}")
        sys.exit(1)
    
    print("Starting Streamlit application...")
    print("The app will open in your default web browser.")
    print("Press Ctrl+C to stop the application.")
    print()
    
    # Run the Streamlit app
    try:
        subprocess.run(["streamlit", "run", app_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit app: {e}")
    except FileNotFoundError:
        print("Streamlit command not found. Please make sure it's installed and in your PATH.")
        print("You can also try running: python -m streamlit run app/predictor.py")

if __name__ == "__main__":
    main()