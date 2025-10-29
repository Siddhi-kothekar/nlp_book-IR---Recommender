@echo off
REM Quick Start Script for Book Recommendation System
REM Windows Batch File

echo ========================================
echo Book Recommendation System - Quick Start
echo ========================================
echo.

REM Check if data file exists
if not exist "data\goodreads_book.csv" (
    echo ERROR: Book data file not found!
    echo Please place your book CSV file at: data\goodreads_book.csv
    echo.
    echo Required columns: Id, Name, Authors, Description
    pause
    exit /b 1
)

REM Check if keywords.csv exists (already processed)
if not exist "data\keywords.csv" (
    echo Processing book data...
    echo This may take a few minutes on first run...
    echo.
    python scripts/process_books.py --input data/goodreads_book.csv --output data/keywords.csv --config config_books.yaml --build-vectors
    echo.
    if errorlevel 1 (
        echo ERROR: Processing failed!
        pause
        exit /b 1
    )
    echo Processing complete!
    echo.
) else (
    echo Found processed data (data/keywords.csv)
    echo Skipping processing step...
    echo.
)

REM Run the web app
echo Starting web application...
echo.
echo The app will open in your browser automatically.
echo To stop the app, press Ctrl+C in this window.
echo.
streamlit run app/book_app.py

