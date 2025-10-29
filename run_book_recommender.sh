#!/bin/bash
# Quick Start Script for Book Recommendation System
# Linux/Mac Shell Script

echo "========================================"
echo "Book Recommendation System - Quick Start"
echo "========================================"
echo ""

# Check if data file exists
if [ ! -f "data/goodreads_book.csv" ]; then
    echo "ERROR: Book data file not found!"
    echo "Please place your book CSV file at: data/goodreads_book.csv"
    echo ""
    echo "Required columns: Id, Name, Authors, Description"
    exit 1
fi

# Check if keywords.csv exists (already processed)
if [ ! -f "data/keywords.csv" ]; then
    echo "Processing book data..."
    echo "This may take a few minutes on first run..."
    echo ""
    python scripts/process_books.py --input data/goodreads_book.csv --output data/keywords.csv --config config_books.yaml --build-vectors
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Processing failed!"
        exit 1
    fi
    
    echo ""
    echo "Processing complete!"
    echo ""
else
    echo "Found processed data (data/keywords.csv)"
    echo "Skipping processing step..."
    echo ""
fi

# Run the web app
echo "Starting web application..."
echo ""
echo "The app will open in your browser automatically."
echo "To stop the app, press Ctrl+C in this window."
echo ""
streamlit run app/book_app.py

