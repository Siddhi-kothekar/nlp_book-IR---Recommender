@echo off
REM Quick Start Script for Amazon Review-Based Recommender
REM Uses BERT + TF-IDF for content-based recommendations

echo ========================================
echo Amazon Review-Based Recommendation System
echo BERT + TF-IDF Content-Based Filtering
echo ========================================
echo.

REM Set environment variables
set TRANSFORMERS_NO_TF=1
set USE_TF=0
set TOKENIZERS_PARALLELISM=false

REM Check if data file exists
if not exist "data\amazon_reviews.csv" (
    echo ERROR: Amazon reviews file not found!
    echo Please place your CSV file at: data\amazon_reviews.csv
    echo.
    echo Required columns:
    echo   - product_name (or itemName/title)
    echo   - review_text (or reviewText/review)
    echo.
    pause
    exit /b 1
)

REM Run the Streamlit app
echo Starting recommendation system...
echo.
echo The app will process your data on first run.
echo This may take a few minutes (BERT model download + processing).
echo.
echo To stop the app, press Ctrl+C
echo.
streamlit run app/amazon_recommender_app.py

