# Getting Started - Book Recommendation System

## 📋 What You Need

1. **Python 3.8+** (check with: `python --version`)
2. **Your book dataset** (CSV file with book information)
3. **Internet connection** (for downloading BERT models on first run)

## 🎯 3 Simple Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Process Your Book Data
```bash
python scripts/process_books.py --input data/goodreads_book.csv --build-vectors
```

**Note:** Replace `data/goodreads_book.csv` with your actual file path.

### Step 3: Run the Web App
```bash
streamlit run app/book_app.py
```

That's it! Open the browser URL shown in terminal.

---

## 📁 Your Book Data Format

Your CSV file needs these columns:

| Column | Required | Description |
|--------|----------|-------------|
| `Id` | ✅ Yes | Unique book ID |
| `Name` | ✅ Yes | Book title |
| `Authors` | ✅ Yes | Author name(s) |
| `Description` | ✅ Yes | Book description/summary |
| `ISBN` | ⚪ Optional | ISBN number |
| `PublishYear` | ⚪ Optional | Year published |
| `Publisher` | ⚪ Optional | Publisher name |
| `Language` | ⚪ Optional | Language code |

**Example CSV:**
```csv
Id,Name,Authors,Description
1000000,Flight from Eden,Kathryn A. Graham,"What could a computer expert, a mercenary with a past..."
1000001,Roommates Again,Kathryn O. Galbraith,"During their stay at Camp Sleep-Away..."
```

**Place your file at:** `data/goodreads_book.csv`

---

## 🚀 Method 1: Quick Start Script (Easiest)

### Windows:
Double-click: **`run_book_recommender.bat`**

### Linux/Mac:
```bash
chmod +x run_book_recommender.sh
./run_book_recommender.sh
```

---

## 🛠️ Method 2: Manual Steps

### 1. Open Terminal/Command Prompt
Navigate to project folder:
```bash
cd C:\Users\HP\OneDrive\Desktop\Nlp-PROJECT
```

### 2. Install Packages
```bash
pip install -r requirements.txt
```
⏱️ Takes 2-5 minutes

### 3. Process Book Data
```bash
python scripts/process_books.py --input data/goodreads_book.csv --output data/keywords.csv --config config_books.yaml --build-vectors
```
⏱️ Takes ~10-20 seconds per 100 books

**What this does:**
- Cleans text (removes HTML, URLs, etc.)
- Extracts keywords using BERT
- Creates similarity matrix
- Saves everything

### 4. Start Web App
```bash
streamlit run app/book_app.py
```

**You'll see:**
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

Click the URL or open it in your browser!

---

## 💻 What You'll See in the App

### Tab 1: 📚 Book Recommendations
- Search or select a book
- Get similar books
- See why they're recommended (keywords)

### Tab 2: 🔍 Information Retrieval  
- Search by keywords
- Find relevant books
- View top keywords per book

---

## ❌ Common Issues

### "No such file: data/goodreads_book.csv"
**Fix:** Place your book CSV in `data/` folder, or update the path:
```bash
python scripts/process_books.py --input path/to/your/file.csv --build-vectors
```

### "Module not found: keybert"
**Fix:** Install dependencies:
```bash
pip install -r requirements.txt
```

### "Processing is very slow"
**Normal!** First run downloads BERT models (~400MB). Subsequent runs are faster.

### "Out of memory"
**Fix:** Process smaller batches or reduce keywords:
Edit `config_books.yaml`: change `keywords_top_n: 10` to `keywords_top_n: 5`

---

## 📊 Expected Processing Times

| Dataset Size | First Run | Subsequent Runs |
|--------------|-----------|-----------------|
| 100 books | ~5 minutes | ~30 seconds |
| 1,000 books | ~30 minutes | ~5 minutes |
| 10,000 books | ~3-4 hours | ~30 minutes |

*First run includes BERT model download*

---

## ✅ Success Indicators

You're ready when you see:
- ✅ `data/keywords.csv` file exists
- ✅ `artifacts/book_vectors.joblib` file exists
- ✅ Web app opens without errors
- ✅ Can search and get recommendations

---

## 🎓 Next Steps After Running

1. **Try different books** - Test the recommendations
2. **Experiment with searches** - Try various keyword queries
3. **Customize settings** - Edit `config_books.yaml`
4. **Read documentation** - See `README_BOOKS.md` for details

---

## 📚 Need More Help?

- **Detailed Guide:** `HOW_TO_RUN.md`
- **Quick Tips:** `QUICKSTART_BOOKS.md`
- **Full Documentation:** `README_BOOKS.md`
- **Code Examples:** `examples/book_recommendation_example.py`

---

**Happy Book Recommending! 📖✨**

