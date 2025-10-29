@echo off
REM Quick Start Script for Amazon Reviews System
REM Windows Batch File

echo ========================================
echo Amazon Reviews IR + Recommendation System
echo ========================================
echo.

REM Set environment variables before Python runs
set TRANSFORMERS_NO_TF=1
set USE_TF=0
set TOKENIZERS_PARALLELISM=false

REM Check if data file exists
if not exist "data\amazon_reviews.csv" (
    echo ERROR: Amazon reviews file not found!
    echo Please place your CSV file at: data\amazon_reviews.csv
    echo.
    pause
    exit /b 1
)

REM Run the web app (it will build index automatically if needed)
echo Starting web application...
echo.
echo The app will automatically build the index on first run.
echo This may take a few minutes...
echo.
echo To stop the app, press Ctrl+C in this window.
echo.
streamlit run app/streamlit_app.py

