# Hostel Price Prediction - Web Applications Launcher
# This PowerShell script starts both Streamlit and Gradio web applications

Write-Host "===================================================="
Write-Host "Hostel Price Prediction - Web Applications Launcher"
Write-Host "===================================================="
Write-Host

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Python not found"
    }
    Write-Host "Python is installed: $pythonVersion"
} catch {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.7 or higher and ensure it's in your system PATH" -ForegroundColor Red
    Write-Host
    pause
    exit 1
}

# Check if required files exist
if (-not (Test-Path "app\predictor.py")) {
    Write-Host "ERROR: Cannot find Streamlit app file 'app\predictor.py'" -ForegroundColor Red
    Write-Host "Please run this script from the project root directory" -ForegroundColor Red
    Write-Host
    pause
    exit 1
}

if (-not (Test-Path "app\gradio_app.py")) {
    Write-Host "ERROR: Cannot find Gradio app file 'app\gradio_app.py'" -ForegroundColor Red
    Write-Host "Please run this script from the project root directory" -ForegroundColor Red
    Write-Host
    pause
    exit 1
}

# Check if required Python packages are installed
Write-Host "Checking for required Python packages..."

try {
    python -c "import streamlit" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: Streamlit is not installed" -ForegroundColor Yellow
        Write-Host "Installing Streamlit..."
        pip install streamlit
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: Failed to install Streamlit" -ForegroundColor Red
            Write-Host "Please install Streamlit manually using: pip install streamlit" -ForegroundColor Red
            Write-Host
            pause
            exit 1
        }
    }
    Write-Host "Streamlit is available"
} catch {
    Write-Host "ERROR: Failed to check Streamlit installation" -ForegroundColor Red
    Write-Host
    pause
    exit 1
}

try {
    python -c "import gradio" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: Gradio is not installed" -ForegroundColor Yellow
        Write-Host "Installing Gradio..."
        pip install gradio
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: Failed to install Gradio" -ForegroundColor Red
            Write-Host "Please install Gradio manually using: pip install gradio" -ForegroundColor Red
            Write-Host
            pause
            exit 1
        }
    }
    Write-Host "Gradio is available"
} catch {
    Write-Host "ERROR: Failed to check Gradio installation" -ForegroundColor Red
    Write-Host
    pause
    exit 1
}

try {
    python -c "import pandas" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: Pandas is not installed" -ForegroundColor Yellow
        Write-Host "Installing Pandas..."
        pip install pandas
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: Failed to install Pandas" -ForegroundColor Red
            Write-Host "Please install Pandas manually using: pip install pandas" -ForegroundColor Red
            Write-Host
            pause
            exit 1
        }
    }
    Write-Host "Pandas is available"
} catch {
    Write-Host "ERROR: Failed to check Pandas installation" -ForegroundColor Red
    Write-Host
    pause
    exit 1
}

try {
    python -c "import joblib" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: Joblib is not installed" -ForegroundColor Yellow
        Write-Host "Installing Joblib..."
        pip install joblib
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: Failed to install Joblib" -ForegroundColor Red
            Write-Host "Please install Joblib manually using: pip install joblib" -ForegroundColor Red
            Write-Host
            pause
            exit 1
        }
    }
    Write-Host "Joblib is available"
} catch {
    Write-Host "ERROR: Failed to check Joblib installation" -ForegroundColor Red
    Write-Host
    pause
    exit 1
}

Write-Host "All required packages are available."
Write-Host

# Start Streamlit application
Write-Host "Starting Streamlit application on port 8506..."
Start-Process -NoNewWindow -FilePath "python" -ArgumentList "-m", "streamlit", "run", "app/predictor.py", "--server.port", "8506"

# Start Gradio application
Write-Host "Starting Gradio application on port 7863..."
Start-Process -NoNewWindow -FilePath "python" -ArgumentList "app/gradio_app.py"

Write-Host
Write-Host "Applications are starting..."
Write-Host

# Wait a moment for applications to start
Write-Host "Waiting for applications to initialize..."
Start-Sleep -Seconds 8

Write-Host "Opening applications in your default browser..."
Start-Process "http://localhost:8506"
Start-Process "http://localhost:7863"

Write-Host
Write-Host "Both applications should now be running in your browser!"
Write-Host
Write-Host "You can access the applications at:"
Write-Host
Write-Host "Streamlit Application: http://localhost:8506"
Write-Host "Gradio Application:   http://localhost:7863"
Write-Host
Write-Host "To stop the applications, close the terminal windows or press Ctrl+C"
Write-Host
Write-Host "NOTE: This launcher window can be closed without affecting the applications."
Write-Host "The applications will continue running until their windows are closed."
Write-Host
Write-Host "Press any key to exit this launcher (applications will continue running)..."
$Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
exit