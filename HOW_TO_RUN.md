# How to Run the Book Recommendation System

## ⚡ Quickest Way (Windows)

**Just double-click:** `run_book_recommender.bat`

This script automatically:
1. Checks if you have book data
2. Processes it if needed
3. Starts the web app

---

## 🚀 Step-by-Step Guide

### Step 1: Install Dependencies

Open your terminal/command prompt and navigate to the project directory:

```bash
cd C:\Users\HP\OneDrive\Desktop\Nlp-PROJECT
```

Install required packages:

```bash
pip install -r requirements.txt
```

**Note:** This may take a few minutes as it downloads BERT models and other dependencies.

### Step 2: Prepare Your Book Data

Place your book dataset CSV file in the `data/` folder. The file should be named `goodreads_book.csv` or update the path in the script.

**Required CSV columns:**
- `Id` - Book ID
- `Name` - Book title
- `Authors` - Author name(s)
- `Description` - Book description/summary
- (Optional) `ISBN`, `PublishYear`, `Publisher`, `Language`

**Example structure:**
```csv
Id,Name,Authors,Description,...
1000000,Flight from Eden,Kathryn A. Graham,"What could a computer expert, a mercenary with...",...
```

### Step 3: Process the Book Data

This step will:
- Clean and preprocess the text
- Extract keywords using KeyBERT (this takes time on first run)
- Build TF-IDF vectors and similarity matrix
- Save everything for fast recommendations

**Run the processing script:**

```bash
python scripts/process_books.py --input data/goodreads_book.csv --output data/keywords.csv --config config_books.yaml --build-vectors
```

**What happens:**
- ✅ Cleans book descriptions (removes HTML, URLs, etc.)
- ✅ Extracts keywords using BERT embeddings (~1-2 sec per book)
- ✅ Creates TF-IDF vectors
- ✅ Computes similarity matrix
- ✅ Saves everything to `artifacts/` folder

**Time estimate:** 
- First run: ~10-20 seconds per 100 books (BERT model downloads automatically)
- Subsequent runs: Much faster (uses cached data)

**Output files created:**
- `data/preprocessed.csv` - Cleaned data
- `data/keywords.csv` - Books with extracted keywords
- `artifacts/book_vectors.joblib` - Saved vectors (for fast loading)
- `artifacts/book_metadata.json` - Book metadata

### Step 4: Run the Web Application

Start the Streamlit web app:

```bash
streamlit run app/book_app.py
```

**What you'll see:**
- Terminal will show: "You can now view your Streamlit app in your browser."
- Browser will open automatically to `http://localhost:8501`
- If not, copy the URL from terminal and paste in your browser

### Step 5: Use the Application

#### Tab 1: Book Recommendations
1. **Select a book** from the dropdown or **type a book name** manually
2. Click **"Get Recommendations"**
3. See similar books with:
   - Similarity scores
   - Author information
   - Common keywords (why they were recommended)

#### Tab 2: Information Retrieval
1. **Enter search keywords** (e.g., "programming python" or "mystery thriller")
2. Click **"Search Books"**
3. See relevant books ranked by relevance
4. View top keywords for each book

---

## 🔄 Alternative: Step-by-Step Processing

If you want to process in separate steps:

### Only Preprocessing:
```bash
python scripts/process_books.py --input data/goodreads_book.csv --output data/preprocessed.csv --skip-keywords --build-vectors=false
```

### Only Keyword Extraction (after preprocessing):
```bash
python scripts/process_books.py --input data/preprocessed.csv --output data/keywords.csv --skip-preprocessing --build-vectors
```

---

## 💻 Python API Usage

Instead of the web app, you can use Python directly:

```python
import pandas as pd
from src.book_recommender import BookRecommender
from src.book_vectorizer import BookVectorizer

# Load processed keywords
books_df = pd.read_csv("data/keywords.csv")

# Load vectorizer (vectors must be built first)
vectorizer = BookVectorizer()
vectorizer.load_vectors("artifacts/book_vectors.joblib")

# Create recommender
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
    "programming python computer science",
    vectorizer, 
    top_k=10
)

for book_name, score in results:
    print(f"{book_name}: {score:.4f}")
```

---

## 🐛 Troubleshooting

### Error: "No module named 'keybert'"
**Solution:** Run `pip install -r requirements.txt` again

### Error: "Book data not found at data/keywords.csv"
**Solution:** 
1. Make sure you ran Step 3 (processing) first
2. Check that `data/keywords.csv` exists
3. Verify your input CSV file path is correct

### Error: "KeyBERT model not found"
**Solution:** 
- First run downloads models automatically
- Ensure you have internet connection
- Wait for download to complete (may take 5-10 minutes)

### Error: "streamlit: command not found"
**Solution:** 
```bash
pip install streamlit
```

### Slow keyword extraction
**This is normal!** 
- First run: BERT models are large (~400MB)
- Processing: ~1-2 seconds per book
- Be patient - results are worth it!

### Memory errors with large datasets
**Solutions:**
1. Process smaller batches
2. Reduce `keywords_top_n` in `config_books.yaml` (e.g., from 10 to 5)
3. Use a smaller KeyBERT model

---

## 📋 Quick Command Reference

```bash
# Install dependencies
pip install -r requirements.txt

# Process books (full pipeline)
python scripts/process_books.py --input data/goodreads_book.csv --build-vectors

# Run web app
streamlit run app/book_app.py

# Stop web app
Press Ctrl+C in terminal
```

---

## ✅ Success Checklist

- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Book CSV file in `data/` folder
- [ ] Processing script completed successfully
- [ ] `data/keywords.csv` file created
- [ ] `artifacts/book_vectors.joblib` file created
- [ ] Web app running without errors
- [ ] Can see book recommendations in browser

---

## 🎯 Example Run Session

```bash
# Terminal session example:

C:\Users\HP\OneDrive\Desktop\Nlp-PROJECT> pip install -r requirements.txt
[... installation messages ...]

C:\Users\HP\OneDrive\Desktop\Nlp-PROJECT> python scripts/process_books.py --input data/goodreads_book.csv --build-vectors

============================================================
STEP 1: Preprocessing Books Data
============================================================
Loading data from data/goodreads_book.csv...
Loaded 1000 books
✓ Processed 850 books

============================================================
STEP 2: Keyword Extraction using KeyBERT
============================================================
Extracting keywords from 850 books...
✓ Extracted keywords for 850 books

============================================================
STEP 3: Building TF-IDF Vectors and Similarity Matrix
============================================================
Vectorizing 850 books...
Vocabulary size: 5234
Computing cosine similarity matrix...
✓ Saved vectors to artifacts/book_vectors.joblib

============================================================
✅ Processing Complete!
============================================================

C:\Users\HP\OneDrive\Desktop\Nlp-PROJECT> streamlit run app/book_app.py

You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

---

## 📞 Need Help?

1. Check `README_BOOKS.md` for detailed documentation
2. See `QUICKSTART_BOOKS.md` for quick tips
3. Review `examples/book_recommendation_example.py` for code examples
4. Check `config_books.yaml` for configuration options

**That's it! You're ready to recommend books!** 📚✨

