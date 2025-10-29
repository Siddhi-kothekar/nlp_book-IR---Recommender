# Content-Based Book Recommendation System

## Overview

This is a **Content-Based Filtering** book recommendation system that uses **NLP techniques** to recommend books based on their content. The system uses:

- **KeyBERT** for keyword extraction from book descriptions using BERT embeddings
- **TF-IDF** for document vectorization
- **Cosine Similarity** for finding similar books

## Architecture

```
Raw Book Data → Text Cleaning → Keyword Extraction (KeyBERT) → TF-IDF Vectorization → Cosine Similarity → Recommendations
```

### Features

1. **Text Cleaning & Preprocessing**
   - Removes URLs, HTML tags, and punctuations
   - Handles missing values
   - Extracts book series information
   - Language detection and imputation

2. **Keyword Extraction**
   - Uses KeyBERT with BERT embeddings to extract semantically relevant keywords
   - Combines description keywords with metadata (author, publisher, series)

3. **Vectorization**
   - TF-IDF vectorization of extracted keywords
   - Filters common and rare words for better recommendations

4. **Recommendation Types**
   - **Similar Books**: Based on content similarity (theme, author, series)
   - **Information Retrieval**: Search books using keywords or queries

## Project Structure

```
Nlp-PROJECT/
├── src/
│   ├── book_processor.py      # Text cleaning and preprocessing
│   ├── book_keywords.py       # Keyword extraction using KeyBERT
│   ├── book_vectorizer.py     # TF-IDF vectorization and similarity
│   └── book_recommender.py    # Recommendation engine
├── app/
│   └── book_app.py            # Streamlit web application
├── scripts/
│   └── process_books.py       # Data processing pipeline
├── examples/
│   └── book_recommendation_example.py  # Usage examples
├── data/
│   ├── goodreads_book.csv     # Input book data (you provide this)
│   ├── preprocessed.csv       # Cleaned data (generated)
│   └── keywords.csv           # Extracted keywords (generated)
├── artifacts/
│   ├── book_vectors.joblib    # Saved TF-IDF vectors
│   └── book_metadata.json     # Book metadata
└── config_books.yaml          # Configuration file
```

## Installation

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Download NLTK Data** (if needed)
```python
import nltk
nltk.download('punkt')
nltk.download('textcat')
```

## Data Format

Your input CSV file should contain the following columns:
- `Id`: Book ID
- `Name`: Book title
- `Authors`: Author name(s)
- `ISBN`: ISBN number
- `PublishYear`: Publication year
- `Publisher`: Publisher name
- `Language`: Language code (optional)
- `Description`: Book description/summary

Example:
```csv
Id,Name,Authors,ISBN,PublishYear,Publisher,Language,Description
1000000,Flight from Eden,Kathryn A. Graham,0595199402,2001,Writer's Showcase Press,,"What could a computer expert, a mercenary with..."
```

## Usage

### Option 1: Complete Pipeline (Recommended)

Run the complete processing pipeline:

```bash
python scripts/process_books.py \
    --input data/goodreads_book.csv \
    --output data/keywords.csv \
    --config config_books.yaml \
    --build-vectors
```

This will:
1. Preprocess the data (clean text, remove duplicates)
2. Extract keywords using KeyBERT
3. Build TF-IDF vectors and similarity matrix
4. Save all results

### Option 2: Step-by-Step Processing

You can also run steps individually:

```bash
# Step 1: Preprocessing only
python scripts/process_books.py \
    --input data/goodreads_book.csv \
    --output data/preprocessed.csv \
    --skip-keywords \
    --skip-preprocessing=False

# Step 2: Keyword extraction only
python scripts/process_books.py \
    --input data/preprocessed.csv \
    --output data/keywords.csv \
    --skip-preprocessing \
    --build-vectors
```

### Option 3: Run Web Application

1. First, ensure you have processed the data:
```bash
python scripts/process_books.py --input data/goodreads_book.csv --build-vectors
```

2. Run the Streamlit app:
```bash
streamlit run app/book_app.py
```

The app provides:
- **Book Recommendations Tab**: Find similar books by book name
- **Information Retrieval Tab**: Search books using keywords

### Option 4: Python Script Usage

See `examples/book_recommendation_example.py` for a complete example:

```python
from src.book_processor import BookProcessor
from src.book_keywords import BookKeywordExtractor
from src.book_vectorizer import BookVectorizer
from src.book_recommender import BookRecommender

# Load and preprocess
processor = BookProcessor()
books_data = processor.process_books(pd.read_csv("data/goodreads_book.csv"))

# Extract keywords
extractor = BookKeywordExtractor()
books_data = extractor.process_books(books_data)

# Vectorize
vectorizer = BookVectorizer()
vectorizer.fit_transform(books_data["keywords"])
similarity_matrix = vectorizer.compute_similarity()

# Create recommender
recommender = BookRecommender(similarity_matrix, books_data["Name"])

# Get recommendations
recommendations = recommender.recommend_similar_books("book name here", n=5)

# Search books
results = recommender.search_books("programming python", vectorizer, top_k=10)
```

## Configuration

Edit `config_books.yaml` to customize:

```yaml
# Keyword extraction
keywords_top_n: 10  # Number of keywords per book
keybert_model: all-MiniLM-L6-v2  # KeyBERT model

# TF-IDF settings
min_df: 3  # Minimum document frequency
max_df: 0.6  # Maximum document frequency

# Recommendations
n_recommendations: 5  # Default number of recommendations
```

## Recommendation Examples

The system can recommend books based on:

1. **Series Information**: Other books in the same series
   - Example: Recommending other books from "Images of America" series

2. **Theme/Genre**: Books with similar themes
   - Example: All programming books regardless of language

3. **Author**: Other works by the same author
   - Example: Recommending other books by "Dean Koontz"

4. **Content Similarity**: Books with similar descriptions/keywords

## Performance

- **Keyword Extraction**: ~1-2 seconds per book (depending on description length)
- **Vectorization**: Fast (TF-IDF is efficient)
- **Similarity Calculation**: O(n²) but computed once and cached
- **Recommendations**: Near-instant (uses precomputed similarity matrix)

## Improvements & Future Work

- Add lemmatization and POS tagging for better keyword quality
- Include genre information in recommendations
- Support for hybrid filtering (collaborative + content-based)
- Advanced reranking using cross-encoders

## References

- [KeyBERT Documentation](https://github.com/MaartenGr/KeyBERT)
- [scikit-learn TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- Goodreads Book Dataset

## License

MIT

