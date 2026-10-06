@echo off
echo Starting Medical X-ray AI System...
echo =====================================
echo.
echo 📊 Loading dependencies...
python -c "import streamlit, tensorflow, numpy, pandas; print('✅ All dependencies loaded successfully')"

if errorlevel 1 (
    echo ❌ Missing dependencies. Please install requirements:
    echo pip install streamlit tensorflow numpy pandas pillow
    pause
    exit /b 1
)

echo.
echo 🚀 Starting Streamlit application...
echo.
echo 🌐 The application will be available at:
echo    http://localhost:8503
echo.
echo ⚠️  Important: Keep this window open while using the app
echo    Press Ctrl+C to stop the application
echo.

python -m streamlit run app.py --server.port 8503 --server.headless false

echo.
echo 🛑 Application stopped.
pause