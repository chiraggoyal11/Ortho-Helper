@echo off
echo 🏥 Medical X-ray AI Classification System
echo ==========================================
echo.

echo Activating virtual environment...
call ".venv\Scripts\activate.bat"

echo.
echo Starting Streamlit application...
echo Open http://localhost:8501 in your browser
echo.
echo Demo Credentials:
echo Doctor: username=doctor, password=medical123
echo Student: username=student, password=learn123
echo.

streamlit run app.py
pause