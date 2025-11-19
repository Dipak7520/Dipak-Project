@echo off
setlocal enabledelayedexpansion

REM Hostel Price Prediction - Web Applications Launcher
REM This batch file starts both Streamlit and Gradio web applications

TITLE Hostel Price Prediction Applications

echo ====================================================
echo Hostel Price Prediction - Web Applications Launcher
echo ====================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.7 or higher and ensure it's in your system PATH
    echo.
    pause
    exit /b 1
)

REM Check if required files exist
if not exist "app\predictor.py" (
    echo ERROR: Cannot find Streamlit app file "app\predictor.py"
    echo Please run this batch file from the project root directory
    echo.
    pause
    exit /b 1
)

if not exist "app\gradio_app.py" (
    echo ERROR: Cannot find Gradio app file "app\gradio_app.py"
    echo Please run this batch file from the project root directory
    echo.
    pause
    exit /b 1
)

REM Check if required Python packages are installed
echo Checking for required Python packages...
python -c "import streamlit" >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: Streamlit is not installed
    echo Installing Streamlit...
    pip install streamlit
    if !errorlevel! neq 0 (
        echo ERROR: Failed to install Streamlit
        echo Please install Streamlit manually using: pip install streamlit
        echo.
        pause
        exit /b 1
    )
)

python -c "import gradio" >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: Gradio is not installed
    echo Installing Gradio...
    pip install gradio
    if !errorlevel! neq 0 (
        echo ERROR: Failed to install Gradio
        echo Please install Gradio manually using: pip install gradio
        echo.
        pause
        exit /b 1
    )
)

python -c "import pandas" >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: Pandas is not installed
    echo Installing Pandas...
    pip install pandas
    if !errorlevel! neq 0 (
        echo ERROR: Failed to install Pandas
        echo Please install Pandas manually using: pip install pandas
        echo.
        pause
        exit /b 1
    )
)

python -c "import joblib" >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: Joblib is not installed
    echo Installing Joblib...
    pip install joblib
    if !errorlevel! neq 0 (
        echo ERROR: Failed to install Joblib
        echo Please install Joblib manually using: pip install joblib
        echo.
        pause
        exit /b 1
    )
)

echo All required packages are available.
echo.

REM Start Streamlit application
echo Starting Streamlit application on port 8506...
start "Streamlit App" /D "%cd%" python -m streamlit run app/predictor.py --server.port 8506 --logger.level info

REM Check if Streamlit started successfully (wait a moment)
timeout /t 3 /nobreak >nul
echo Streamlit application start command issued.
echo.

REM Start Gradio application
echo Starting Gradio application on port 7863...
start "Gradio App" /D "%cd%" python app/gradio_app.py

REM Check if Gradio started successfully (wait a moment)
timeout /t 3 /nobreak >nul
echo Gradio application start command issued.
echo.

echo Applications are starting...
echo.

echo You can access the applications at:
echo.
echo Streamlit Application: http://localhost:8506
echo Gradio Application:   http://localhost:7863
echo.

REM Wait a moment for applications to fully start
echo Waiting for applications to initialize...
timeout /t 8 /nobreak >nul

REM Open applications in browser
echo Opening applications in your default browser...
start http://localhost:8506
start http://localhost:7863

echo.
echo Both applications should now be running in your browser!
echo.
echo To stop the applications:
echo 1. Close the terminal windows that opened for each application, OR
echo 2. Press Ctrl+C in each terminal window
echo.
echo NOTE: This launcher window can be closed without affecting the applications.
echo The applications will continue running until their windows are closed.
echo.

echo Press any key to exit this launcher...
pause >nul
exit