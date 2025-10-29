# Amazon Reviews - Information Retrieval & Recommendation System

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Place Your Data
Put `amazon_reviews.csv` in the `data/` folder.

### 3. Run the App
```bash
streamlit run app/streamlit_app.py
```

Or Windows:
```bash
run_amazon_reviews.bat
```

The app will automatically build the index on first run.

---

## 📋 Required CSV Format

Your `amazon_reviews.csv` should have these columns:

| Column | Required | Description |
|--------|----------|-------------|
| `asin` or `itemName` | ✅ Yes | Product ID |
| `reviewText` or `review` | ✅ Yes | Review text |
| `rating` or `overall` | ⚪ Optional | Rating (1-5) |
| `itemName` or `title` | ⚪ Optional | Product name |
| `userName` or `user` | ⚪ Optional | Reviewer name |

**Example:**
```csv
asin,itemName,rating,reviewText,userName
B001,Great Product,5,"This product is amazing!",john_doe
B002,Okay Product,3,"It's okay, nothing special",jane_smith
```

---

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
model_name: sentence-transformers/all-mpnet-base-v2  # Embedding model
data_path: data/amazon_reviews.csv                  # Your data file
top_k: 50                                           # Reviews to retrieve
recommendations: 10                                  # Products to recommend
min_rating: 4                                        # Minimum rating filter
```

---

## 🔍 Features

1. **Information Retrieval**
   - Search reviews by query
   - Find relevant reviews using semantic search
   - Filter by sentiment and rating

2. **Product Recommendations**
   - Get product recommendations based on search
   - Combines similarity, sentiment, and ratings
   - Shows top products matching your query

---

## 🛠️ Troubleshooting

### TensorFlow Error?
Fixed! Environment variables are set automatically.

### "No module named 'src'"
Run from project root directory.

### Slow First Run?
Normal! Building embeddings takes time. Subsequent runs are fast.

---

## 📁 Essential Files

- `src/` - Core modules
- `app/streamlit_app.py` - Web interface
- `config.yaml` - Configuration
- `data/amazon_reviews.csv` - Your data

See `ESSENTIAL_FILES.md` for full list.

