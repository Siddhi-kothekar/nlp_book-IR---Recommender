# Quick Start Guide: Book Recommendation System

## 🚀 Quick Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data

Place your book dataset (CSV format) in the `data/` folder. Required columns:
- `Id`, `Name`, `Authors`, `Description`
- Optional: `ISBN`, `PublishYear`, `Publisher`, `Language`

**Example file:** `data/goodreads_book.csv`

### 3. Run the Processing Pipeline

Process your data in one command:

```bash
python scripts/process_books.py \
    --input data/goodreads_book.csv \
    --output data/keywords.csv \
    --config config_books.yaml \
    --build-vectors
```

This will:
- ✅ Clean and preprocess book descriptions
- ✅ Extract keywords using KeyBERT (BERT embeddings)
- ✅ Build TF-IDF vectors and similarity matrix
- ✅ Save everything for fast recommendations

**Time estimate:** ~10-20 seconds per 100 books (depends on CPU)

### 4. Run the Web App

```bash
streamlit run app/book_app.py
```

Open your browser to the URL shown (usually `http://localhost:8501`)

## 📚 Using the System

### Web Interface Features

**Tab 1: Book Recommendations**
- Select or type a book name
- Get top N similar books
- See why books were recommended (common keywords)

**Tab 2: Information Retrieval**
- Search with keywords: "programming python"
- Get relevant books ranked by relevance
- View top keywords for each book

### Python API Usage

```python
from src.book_recommender import BookRecommender
from src.book_vectorizer import BookVectorizer
import pandas as pd

# Load processed data
books_df = pd.read_csv("data/keywords.csv")

# Load vectorizer (if already built)
vectorizer = BookVectorizer()
vectorizer.load_vectors("artifacts/book_vectors.joblib")

# Load recommender (similarity matrix should be loaded with vectorizer)
recommender = BookRecommender(
    vectorizer.similarity_matrix,
    books_df["Name"]
)

# Get recommendations
recommendations = recommender.recommend_similar_books(
    "the practice of programming", 
    n=5
)

for book_name, score in recommendations:
    print(f"{book_name}: {score:.4f}")

# Search books
results = recommender.search_books(
    "mystery thriller detective", 
    vectorizer, 
    top_k=10
)
```

## 🛠️ Troubleshooting

### Error: "Book data not found"
- Make sure you've run the processing pipeline first
- Check that `data/keywords.csv` exists

### Error: "KeyBERT model not found"
- The first run will download the model automatically
- Ensure you have internet connection

### Slow keyword extraction?
- This is normal for first run (BERT models are large)
- Subsequent runs use cached models
- Consider using a faster model: `all-MiniLM-L6-v2` (default)

### Memory issues with large datasets?
- Process in batches
- Reduce `keywords_top_n` in config
- Use a smaller KeyBERT model

## 📝 Example Workflow

1. **Start with sample data** (if you have `data/goodreads_book.csv`):
```bash
python scripts/process_books.py --input data/goodreads_book.csv --build-vectors
```

2. **Try the web app**:
```bash
streamlit run app/book_app.py
```

3. **Test recommendations**: Search for "programming" or select a book name

4. **Customize**: Edit `config_books.yaml` to adjust settings

## 🎯 What You Get

- **Content-based recommendations** based on:
  - Book themes and genres
  - Author information
  - Series information
  - Description keywords

- **Information retrieval** for:
  - Finding books by topic
  - Semantic search over descriptions
  - Keyword-based discovery

## 📖 Next Steps

- See `README_BOOKS.md` for detailed documentation
- Check `examples/book_recommendation_example.py` for code examples
- Customize `config_books.yaml` for your needs

Enjoy recommending books! 📚

