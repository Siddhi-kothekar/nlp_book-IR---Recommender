# NLP-based Information Retrieval and Recommendation System

A comprehensive dual-purpose recommendation system that provides content-based recommendations for both **books** and **Amazon products** using advanced Natural Language Processing (NLP) techniques.

## 📋 Project Description

This project implements two distinct recommendation systems:

1. **Book Recommendation System**: Content-based filtering using KeyBERT for keyword extraction, TF-IDF vectorization, and cosine similarity to recommend books based on their descriptions, authors, and themes.


### Key Features

- **Semantic Search**: Uses transformer-based embeddings (BERT, Sentence Transformers) for semantic understanding
- **Content-Based Filtering**: Recommends items based on content similarity rather than user behavior
- **Sentiment Analysis**: Integrates VADER sentiment analysis for Amazon reviews
- **Fast Retrieval**: FAISS indexing for efficient similarity search
- **Interactive UI**: Streamlit-based web applications for both systems
- **Scalable**: Handles large datasets with batch processing

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+** (Python 3.10+ recommended)
- **pip** package manager
- **Internet connection** (for downloading pre-trained models on first run)
- **Git** (for cloning the repository)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Siddhi-kothekar/nlp_book-IR---Recommender.git
cd nlp_book-IR---Recommender
```

### Step 2: Create and Activate Virtual Environment

#### Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux/Mac:
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: This installation may take 5-10 minutes as it downloads:
- Sentence Transformers models (~400MB)
- KeyBERT dependencies
- Other required libraries

### Step 4: Download NLTK Data (if needed)

```python
import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('textcat')
```

Or run:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('vader_lexicon'); nltk.download('textcat')"
```

### Step 5: Prepare Your Data

#### For Book Recommendations:
Place your book dataset CSV file in the `data/` folder. The file should contain:
- Required columns: `Id`, `Name`, `Authors`, `Description`
- Optional columns: `ISBN`, `PublishYear`, `Publisher`, `Language`

#### For Amazon Product Recommendations:
Place your Amazon reviews CSV file in the `data/` folder. The file should contain:
- Required columns: `asin` (or `itemName`), `reviewText` (or `review`)
- Optional columns: `rating` (or `overall`), `title` (or `itemName`), `userName`

### Step 6: Run the Applications

#### Book Recommendation System:
```bash
# Process the book data first
python scripts/process_books.py --input data/goodreads_book.csv --build-vectors

# Run the web app
streamlit run app/book_app.py
```

#### Amazon Product Recommendation System:
```bash
# The app will automatically build the index on first run
streamlit run app/streamlit_app.py
```

Or use the quick start scripts:
- **Windows**: Double-click `run_book_recommender.bat` or `run_amazon_reviews.bat`
- **Linux/Mac**: `chmod +x run_book_recommender.sh && ./run_book_recommender.sh`

---

## 📊 Datasets

### Book Dataset

**Source**: Goodreads Book Dataset (or similar book metadata)

**Format**: CSV file with the following structure:

| Column | Required | Description |
|--------|----------|-------------|
| `Id` | ✅ Yes | Unique book identifier |
| `Name` | ✅ Yes | Book title |
| `Authors` | ✅ Yes | Author name(s) |
| `Description` | ✅ Yes | Book description/summary |
| `ISBN` | ⚪ Optional | ISBN number |
| `PublishYear` | ⚪ Optional | Publication year |
| `Publisher` | ⚪ Optional | Publisher name |
| `Language` | ⚪ Optional | Language code |

**Example**:
```csv
Id,Name,Authors,Description
1000000,Flight from Eden,Kathryn A. Graham,"What could a computer expert, a mercenary with a past..."
1000001,Roommates Again,Kathryn O. Galbraith,"During their stay at Camp Sleep-Away..."
```

**Preprocessing Steps**:
1. Text cleaning (removes HTML tags, URLs, special characters)
2. Missing value handling
3. Language detection and filtering
4. Keyword extraction using KeyBERT (BERT embeddings)
5. TF-IDF vectorization
6. Cosine similarity matrix computation

**Size**: The system can handle datasets of various sizes:
- Small: 100-1,000 books (~30 seconds processing)
- Medium: 1,000-10,000 books (~5-30 minutes)
- Large: 10,000+ books (may require batch processing)

### Amazon Reviews Dataset

**Source**: Amazon Product Reviews Dataset (or similar review data)

**Format**: CSV file with the following structure:

| Column | Required | Description |
|--------|----------|-------------|
| `asin` | ✅ Yes | Product ID (Amazon Standard Identification Number) |
| `reviewText` | ✅ Yes | Review text content |
| `rating` or `overall` | ⚪ Optional | Numeric rating (1-5) |
| `title` or `itemName` | ⚪ Optional | Product name |
| `userName` | ⚪ Optional | Reviewer name |

**Example**:
```csv
asin,itemName,rating,reviewText,userName
B001,Great Product,5,"This product is amazing! Highly recommend.",john_doe
B002,Okay Product,3,"It's okay, nothing special.",jane_smith
```

**Preprocessing Steps**:
1. Text cleaning (removes URLs, HTML tags, punctuation)
2. Review length filtering (max 512 tokens)
3. Sentiment analysis using VADER
4. Embedding generation using Sentence Transformers
5. FAISS index construction for fast retrieval

**Size**: The system supports:
- Small: 1,000-10,000 reviews
- Medium: 10,000-100,000 reviews
- Large: 100,000+ reviews (with batch processing)

**Note**: Large dataset files (>100MB) are excluded from this repository due to GitHub's file size limits. Users need to provide their own datasets. Place your CSV files in the `data/` directory. The `.gitignore` file is configured to exclude large data files automatically.

---

## 📁 Directory Structure

```
nlp_book-IR---Recommender/
│
├── src/                          # Core source modules
│   ├── __init__.py
│   ├── book_processor.py        # Book text cleaning and preprocessing
│   ├── book_keywords.py         # KeyBERT keyword extraction for books
│   ├── book_vectorizer.py       # TF-IDF vectorization for books
│   ├── book_recommender.py      # Book recommendation engine
│   ├── data.py                  # Data loading utilities
│   ├── embeddings.py            # Sentence transformer embeddings
│   ├── indexer.py               # FAISS indexing for Amazon reviews
│   ├── recommender.py           # Amazon product recommendation logic
│   ├── reranker.py              # Cross-encoder reranking (optional)
│   └── sentiment.py             # VADER sentiment analysis
│
├── app/                          # Streamlit web applications
│   ├── book_app.py              # Book recommendation UI
│   ├── streamlit_app.py         # Amazon reviews UI
│   └── amazon_recommender_app.py # Alternative Amazon UI
│
├── scripts/                      # Processing scripts
│   ├── build_index.py           # Build FAISS index for Amazon reviews
│   └── process_books.py         # Book data processing pipeline
│
├── examples/                     # Usage examples
│   └── book_recommendation_example.py
│
├── data/                         # Data directory (user-provided)
│   └── keywords.csv             # Processed keywords (generated)
│   # Note: Large CSV files are excluded from git (see .gitignore)
│   # Users should add their own book/Amazon review datasets here
│
├── artifacts/                    # Saved models and indices
│   ├── book_vectors.joblib      # Saved TF-IDF vectors for books
│   ├── book_metadata.json       # Book metadata
│   ├── reviews.index.faiss      # FAISS index for reviews
│   └── reviews.meta.json        # Review metadata
│
├── config.yaml                   # Configuration for Amazon system
├── config_books.yaml             # Configuration for book system
├── requirements.txt              # Python dependencies
│
├── amazon_recommender.py         # Standalone Amazon recommender
│
├── run_book_recommender.bat     # Windows quick start (books)
├── run_book_recommender.sh      # Linux/Mac quick start (books)
├── run_amazon_reviews.bat        # Windows quick start (Amazon)
│
└── README.md                     # This file
```

---

## 💡 Usage Examples

### Book Recommendation System

#### Input Example:
```python
# User searches for: "The Practice of Programming"
# Or selects from dropdown: "The Practice of Programming"
```

#### Expected Output:
```
Recommended Books:
1. The Practice of Programming (Similarity: 0.95)
   Author: Brian W. Kernighan, Rob Pike
   Common Keywords: programming, software, computer science, algorithms

2. Code Complete (Similarity: 0.87)
   Author: Steve McConnell
   Common Keywords: programming, software development, best practices

3. Clean Code (Similarity: 0.82)
   Author: Robert C. Martin
   Common Keywords: programming, code quality, software engineering
```

#### Information Retrieval Example:
**Input Query**: `"programming python computer science"`

**Output**:
```
Search Results:
1. Python Programming: An Introduction to Computer Science (Relevance: 0.92)
   Keywords: python, programming, computer science, algorithms

2. Learning Python (Relevance: 0.88)
   Keywords: python, programming, learning, tutorial
```

### Amazon Product Recommendation System

#### Input Example:
```python
# User query: "durable wireless headphones with good battery life"
```

#### Expected Output:
```
Top Recommended Products:

1. Sony WH-1000XM4 Wireless Headphones
   Relevance Score: 0.89
   Average Rating: 4.6/5.0
   Sentiment: 0.75 (Positive)
   Matching Reviews: 45 reviews found
   
2. Bose QuietComfort 35 II
   Relevance Score: 0.85
   Average Rating: 4.5/5.0
   Sentiment: 0.72 (Positive)
   Matching Reviews: 38 reviews found
```

#### Review Retrieval Example:
**Input Query**: `"battery life excellent sound quality"`

**Output**:
```
Retrieved Reviews (Top 5):

1. "Excellent battery life! Lasts 30+ hours on a single charge. Sound quality is amazing..."
   Product: Sony WH-1000XM4
   Rating: 5/5 | Sentiment: 0.85

2. "Great headphones, battery lasts all day. Sound is crisp and clear..."
   Product: Bose QuietComfort 35 II
   Rating: 5/5 | Sentiment: 0.78
```

---

## 🛠️ Technologies and Libraries

### Programming Languages
- **Python 3.8+** (Primary language)

### Core Libraries

#### Natural Language Processing
- **sentence-transformers** (>=3.0.0): Pre-trained transformer models for embeddings
  - Models used: `all-MiniLM-L6-v2`, `all-mpnet-base-v2`
- **keybert** (>=0.5.1): Keyword extraction using BERT embeddings
- **nltk** (>=3.9): Natural language toolkit for text processing
  - VADER sentiment analyzer
  - Tokenization and text categorization

#### Machine Learning & Data Processing
- **scikit-learn** (>=1.3.0): TF-IDF vectorization, cosine similarity
- **pandas** (>=2.1.0): Data manipulation and analysis
- **numpy** (>=1.26.0): Numerical computations

#### Information Retrieval
- **faiss-cpu** (>=1.7.4): Efficient similarity search and clustering
  - Used for fast vector similarity search in Amazon reviews

#### Web Framework
- **streamlit** (>=1.36.0): Interactive web application framework

#### Utilities
- **pyyaml** (>=6.0.1): Configuration file parsing
- **tqdm** (>=4.66.0): Progress bars for long-running operations
- **joblib** (>=1.3.0): Model serialization and parallel processing

### Tools and Frameworks
- **Git**: Version control
- **FAISS**: Facebook AI Similarity Search for efficient vector operations
- **Hugging Face Transformers**: Pre-trained transformer models
- **Streamlit**: Web application framework

### Architecture Components

#### Book Recommendation System:
- **KeyBERT**: BERT-based keyword extraction
- **TF-IDF**: Term Frequency-Inverse Document Frequency vectorization
- **Cosine Similarity**: Content-based similarity computation

#### Amazon Recommendation System:
- **Sentence Transformers**: Semantic embeddings for reviews
- **FAISS**: Fast approximate nearest neighbor search
- **VADER**: Sentiment analysis
- **Cross-Encoder Reranking** (optional): Advanced reranking using cross-encoders

---

## 🔧 Configuration

### Book System Configuration (`config_books.yaml`)

```yaml
# Model Configuration
keybert_model: all-MiniLM-L6-v2
sentence_transformer: sentence-transformers/all-MiniLM-L6-v2

# Data Processing
min_description_word_count: 3
keywords_top_n: 10

# TF-IDF Vectorization
min_df: 3
max_df: 0.6

# Recommendation Settings
n_recommendations: 5
ir_top_k: 10
```

### Amazon System Configuration (`config.yaml`)

```yaml
model_name: sentence-transformers/all-mpnet-base-v2
reranker_model: cross-encoder/ms-marco-MiniLM-L-6-v2
artifacts_dir: artifacts
data_path: data/amazon_reviews.csv

top_k: 50
recommendations: 10
batch_size: 256
max_review_length: 512
sentiment_threshold: 0.1
min_rating: 4
```

---

## 📈 Performance Considerations

### Processing Times

**Book System**:
- First run (with model download): ~10-20 seconds per 100 books
- Subsequent runs: ~1-2 seconds per 100 books
- Keyword extraction: ~1-2 seconds per book

**Amazon System**:
- Index building: ~1-2 minutes per 10,000 reviews
- Query retrieval: <100ms per query
- Embedding generation: ~0.5 seconds per review (batch processing)

### Memory Requirements

- **Minimum**: 4GB RAM
- **Recommended**: 8GB+ RAM for large datasets
- **Large datasets**: May require 16GB+ RAM

### Optimization Tips

1. Use batch processing for large datasets
2. Reduce `keywords_top_n` in book config for faster processing
3. Use smaller embedding models for faster inference
4. Enable FAISS GPU support for faster search (requires `faiss-gpu`)

---

## 🎯 Features in Detail

### Book Recommendation Features

1. **Content-Based Filtering**: Recommends books based on description similarity
2. **Author-Based Recommendations**: Finds books by the same author
3. **Series Detection**: Recommends other books in the same series
4. **Keyword-Based Search**: Information retrieval using extracted keywords
5. **Similarity Scoring**: Cosine similarity-based ranking

### Amazon Recommendation Features

1. **Semantic Search**: Finds relevant reviews using semantic understanding
2. **Sentiment-Aware Recommendations**: Weights recommendations by sentiment scores
3. **Rating Integration**: Combines similarity, sentiment, and ratings
4. **Product Aggregation**: Groups reviews by product and ranks products
5. **Fast Retrieval**: FAISS-based efficient similarity search

---

## 📸 Screenshots and Visualizations

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Book Recommendation System                │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Raw Book Data → Text Cleaning → Keyword Extraction         │
│       (KeyBERT) → TF-IDF Vectorization → Cosine Similarity  │
│       → Recommendations                                       │
│                                                               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              Amazon Product Recommendation System             │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Reviews → Text Cleaning → Embeddings (Sentence Transformers)│
│       → FAISS Index → Sentiment Analysis (VADER)            │
│       → Product Aggregation → Recommendations                │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Input CSV
    ↓
Preprocessing (Text Cleaning, Normalization)
    ↓
Feature Extraction (Keywords/Embeddings)
    ↓
Vectorization (TF-IDF/Embeddings)
    ↓
Similarity Computation
    ↓
Recommendation Generation
    ↓
Web UI Display
```

---

## 🤝 Collaborators

**Current Collaborators:**
- Siddhi-kothekar (Repository Owner)

**To Add Collaborators:**

The repository owner needs to add collaborators through GitHub's web interface:

1. Go to the repository on GitHub: https://github.com/Siddhi-kothekar/nlp_book-IR---Recommender
2. Click on **Settings** (in the repository navigation bar)
3. Click on **Collaborators** in the left sidebar
4. Click **Add people** button
5. Search for and add:
   - `surajamit` or `Amit Purushottam Pimpalkar`
   - Any other collaborators as needed
6. Collaborators will receive an email invitation to accept

**Note**: Only the repository owner can add collaborators. Please add `surajamit` or `Amit Purushottam Pimpalkar` as requested.

---

## 📝 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **Hugging Face** for pre-trained transformer models
- **Sentence Transformers** library for semantic embeddings
- **KeyBERT** for keyword extraction
- **FAISS** for efficient similarity search
- **Streamlit** for the web framework
- **Goodreads** and **Amazon** for dataset inspiration

---

## 📚 Additional Documentation

- `README_BOOKS.md`: Detailed documentation for book recommendation system
- `README_AMAZON.md`: Detailed documentation for Amazon recommendation system
- `GETTING_STARTED.md`: Quick start guide
- `HOW_TO_RUN.md`: Step-by-step running instructions
- `QUICKSTART_BOOKS.md`: Quick tips for book system
- `ESSENTIAL_FILES.md`: List of essential project files

---

## 🐛 Troubleshooting

### Common Issues

1. **"Module not found" errors**
   - Solution: Run `pip install -r requirements.txt`

2. **"NLTK data not found"**
   - Solution: Run the NLTK download commands in Step 4

3. **"Out of memory" errors**
   - Solution: Process smaller batches or reduce dataset size

4. **Slow processing**
   - Solution: Normal on first run (model downloads). Subsequent runs are faster.

5. **"File not found" errors**
   - Solution: Ensure data files are in the `data/` directory

---

## 🔮 Future Enhancements

- [ ] Hybrid filtering (collaborative + content-based)
- [ ] Advanced reranking with cross-encoders
- [ ] Multi-language support
- [ ] Real-time recommendation updates
- [ ] User preference learning
- [ ] API endpoints for integration
- [ ] Docker containerization
- [ ] Cloud deployment guides

---

## 📧 Contact

For questions or issues, please open an issue on the GitHub repository.

---

**Happy Recommending! 📚✨**
