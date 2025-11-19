#!/usr/bin/env python3
"""
Main script to run the entire hostel price prediction project workflow
"""

import os
import sys
import subprocess
import argparse

def run_command(command, description=""):
    """
    Run a command and handle errors
    
    Args:
        command (str): Command to run
        description (str): Description of what the command does
    """
    print(f"\n{'='*50}")
    if description:
        print(f"Running: {description}")
    print(f"Command: {command}")
    print('='*50)
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run the hostel price prediction project")
    parser.add_argument("--steps", nargs="+", 
                       choices=["generate_data", "train_models", "run_eda", "run_streamlit", "run_gradio"],
                       default=["train_models"],
                       help="Steps to run (default: train_models)")
    parser.add_argument("--price_multiplier", type=float, default=1.0,
                       help="Price multiplier factor (default: 1.0)")
    
    args = parser.parse_args()
    
    print("🏨 Hostel Price Prediction Project")
    print("Starting project workflow...\n")
    
    # Change to project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    
    # Step 1: Generate data with price multiplier (if requested)
    if "generate_data" in args.steps:
        print(f"Step 1: Generating extended dataset with price multiplier {args.price_multiplier}...")
        if not run_command(f"python src/data_generator.py {args.price_multiplier}", "Generate synthetic hostel data"):
            print("Failed to generate data. Exiting.")
            sys.exit(1)
    
    # Step 2: Train models (if requested)
    if "train_models" in args.steps:
        print("Step 2: Training models...")
        if not run_command(f"python src/main.py {args.price_multiplier}", "Train and evaluate models"):
            print("Failed to train models. Exiting.")
            sys.exit(1)
    
    # Step 3: Run EDA notebook (if requested)
    if "run_eda" in args.steps:
        print("Step 3: Running EDA notebook...")
        print("Please run 'jupyter notebook notebooks/eda_notebook.ipynb' manually to explore the data.")
    
    # Step 4: Run Streamlit app (if requested)
    if "run_streamlit" in args.steps:
        print("Step 4: Starting Streamlit application...")
        print("To run the Streamlit app, execute: streamlit run app/predictor.py")
    
    # Step 5: Run Gradio app (if requested)
    if "run_gradio" in args.steps:
        print("Step 5: Starting Gradio application...")
        print("To run the Gradio app, execute: python app/gradio_app.py")
    
    print("\n✅ Project workflow completed!")
    print("\nTo explore the results:")
    print("- Check the 'results' folder for visualizations and model comparison")
    print("- Check the 'models' folder for saved models")
    print("- Run 'streamlit run app/predictor.py' for the web app")

if __name__ == "__main__":
    main()