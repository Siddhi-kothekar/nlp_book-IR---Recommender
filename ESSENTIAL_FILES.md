# Essential Files for Amazon Reviews System

## ✅ Required Files (Keep These)

### Core Source Files (`src/`)
- `src/data.py` - Data loading and cleaning
- `src/embeddings.py` - Sentence transformer embeddings
- `src/indexer.py` - FAISS/NumPy indexing
- `src/recommender.py` - Product recommendation logic
- `src/sentiment.py` - Sentiment analysis

### Application (`app/`)
- `app/streamlit_app.py` - Main web application

### Scripts (`scripts/`)
- `scripts/build_index.py` - Build index from CSV (optional, app can do this)

### Configuration
- `config.yaml` - Configuration file
- `requirements.txt` - Python dependencies

### Data
- `data/amazon_reviews.csv` - Your Amazon reviews dataset

### Optional
- `run_amazon_reviews.bat` - Quick start script (Windows)

---

## ❌ Book-Related Files (Can Be Removed)

These are for the **book recommendation system**, not needed for Amazon Reviews:

### Book Source Files (`src/`)
- `src/book_processor.py`
- `src/book_keywords.py`
- `src/book_vectorizer.py`
- `src/book_recommender.py`

### Book Application
- `app/book_app.py`

### Book Scripts
- `scripts/process_books.py`

### Book Documentation
- `README_BOOKS.md`
- `QUICKSTART_BOOKS.md`
- `HOW_TO_RUN.md` (book-specific)
- `GETTING_STARTED.md` (book-specific)
- `IMPLEMENTATION_SUMMARY.md` (book-specific)

### Book Configuration
- `config_books.yaml`

### Book Scripts
- `run_book_recommender.bat`
- `run_book_recommender.sh`
- `examples/book_recommendation_example.py`

### Other
- `colab_amazon_ir.ipynb` (optional, for Colab)

---

## 🗂️ Minimal Project Structure for Amazon Reviews

```
Nlp-PROJECT/
├── src/
│   ├── __init__.py
│   ├── data.py              ✅ Required
│   ├── embeddings.py        ✅ Required
│   ├── indexer.py           ✅ Required
│   ├── recommender.py       ✅ Required
│   └── sentiment.py         ✅ Required
├── app/
│   └── streamlit_app.py     ✅ Required
├── scripts/
│   └── build_index.py       ✅ Optional (app can build automatically)
├── artifacts/               (created automatically)
├── data/
│   └── amazon_reviews.csv   ✅ Your data file
├── config.yaml              ✅ Required
├── requirements.txt         ✅ Required
└── run_amazon_reviews.bat   ⚪ Optional (convenience script)
```

---

## 🚀 Quick Start (Amazon Reviews Only)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Place your data:**
   - Put `amazon_reviews.csv` in `data/` folder

3. **Run the app:**
   ```bash
   streamlit run app/streamlit_app.py
   ```

   Or on Windows:
   ```bash
   run_amazon_reviews.bat
   ```

That's it! The app will automatically:
- Load `data/amazon_reviews.csv`
- Build embeddings and index (first time only)
- Start serving recommendations

---

## 📝 What Each Essential File Does

### `src/data.py`
- Loads CSV files
- Maps columns to standard format
- Cleans review text

### `src/embeddings.py`
- Creates embeddings using sentence transformers
- Normalizes vectors for cosine similarity

### `src/indexer.py`
- Creates FAISS or NumPy index for fast similarity search
- Handles vector storage and retrieval

### `src/recommender.py`
- Aggregates review scores to product recommendations
- Combines similarity, sentiment, and ratings

### `src/sentiment.py`
- Analyzes sentiment of reviews using VADER

### `app/streamlit_app.py`
- Web interface for search and recommendations
- Handles all user interactions

### `config.yaml`
- Configuration: model names, file paths, thresholds
- Adjust settings here

