# Amazon Review-Based Recommendation System

## 🎯 Overview

Content-Based Product Recommendation System using:
- **BERT** (via KeyBERT) for keyword extraction from reviews
- **TF-IDF** for document vectorization
- **Cosine Similarity** for finding similar products

---

## 🚀 Quick Start

### Option 1: Streamlit Web App (Recommended)

```bash
# Windows
run_amazon_recommender.bat

# Or manually
streamlit run app/amazon_recommender_app.py
```

### Option 2: Python Script

```bash
python amazon_recommender.py
```

---

## 📋 Requirements

### CSV File Format

Your `data/amazon_reviews.csv` should have:

| Column | Required | Alternative Names |
|--------|----------|-------------------|
| `review_text` | ✅ Yes | `reviewText`, `review` |
| `product_name` | ✅ Yes | `itemName`, `title` |
| `product_id` | ⚪ Optional | `asin` |

**Example CSV:**
```csv
product_id,product_name,review_text,rating
B001,Wireless Mouse,"Great quality and battery life",5
B002,Gaming Keyboard,"Awesome feel but noisy",4
B003,USB-C Cable,"Durable and fast charging",5
```

---

## 🔧 How It Works

### Step 1: Data Loading
- Loads CSV file
- Maps column names automatically
- Cleans review text (removes URLs, HTML, punctuation)

### Step 2: Keyword Extraction (BERT)
- Uses KeyBERT to extract semantically relevant keywords
- Combines keywords from all reviews per product
- Creates product "profiles" based on review content

### Step 3: TF-IDF Vectorization
- Converts keyword text to numerical vectors
- Filters common and rare words

### Step 4: Cosine Similarity
- Computes similarity between all products
- Uses precomputed similarity matrix for fast recommendations

### Step 5: Recommendations
- Finds products with highest similarity scores
- Returns top N similar products

---

## 📖 Usage Examples

### Python Script Usage

```python
from amazon_recommender import *

# Load and process
df = load_and_preprocess_data("data/amazon_reviews.csv", sample_size=1000)
df = extract_keywords_reviews(df)
product_df = aggregate_keywords_by_product(df)
tfidf, tfidf_matrix, cosine_sim = build_recommendation_system(product_df)

# Get recommendations
recommendations = recommend_products(
    product_df, 
    cosine_sim, 
    "Apple AirPods Pro", 
    n=5
)

# Display results
for idx, row in recommendations.iterrows():
    print(f"{row['product_name']}: {row['similarity_score']:.4f}")
```

### Streamlit App Usage

1. Place your CSV at `data/amazon_reviews.csv`
2. Click "Load & Process Data"
3. Select or type a product name
4. Click "Get Recommendations"
5. View similar products with similarity scores

---

## ⚙️ Configuration

### In `amazon_recommender.py`:

```python
# Sample size (None = use all data)
SAMPLE_SIZE = None  # or 1000 for testing

# Keywords to extract per review
top_n = 10

# TF-IDF parameters
min_df = 2  # Minimum document frequency
max_df = 0.7  # Maximum document frequency
```

### In Streamlit App:

- **Sample Data**: Toggle to use subset for faster processing
- **Sample Size**: Slider to control subset size
- **Number of Recommendations**: How many products to return

---

## 📊 Output Example

```
Finding products similar to: 'Apple AirPods Pro'

Top 5 Recommendations:
------------------------------------------------------------
1. Bose QuietComfort Earbuds
   Similarity: 0.8234
   Sample Review: Great noise cancellation...

2. Sony WF-1000XM4
   Similarity: 0.7891
   Sample Review: Premium sound and battery...

3. Samsung Galaxy Buds 2
   Similarity: 0.7654
   Sample Review: Compact and comfortable...

4. Beats Studio Buds
   Similarity: 0.7432
   Sample Review: Great bass response...

5. JBL Tune 230NC
   Similarity: 0.7123
   Sample Review: Best budget noise-canceling...
```

---

## 🗂️ Project Structure

```
Nlp-PROJECT/
├── amazon_recommender.py           # Main script
├── app/
│   └── amazon_recommender_app.py  # Streamlit web app
├── data/
│   └── amazon_reviews.csv         # Your data file
│   └── Processed_Reviews.csv      # Generated (processed data)
├── run_amazon_recommender.bat     # Quick start (Windows)
└── requirements.txt               # Dependencies
```

---

## ⏱️ Performance Notes

### First Run:
- **BERT Model Download**: ~5-10 minutes (one-time download)
- **Keyword Extraction**: ~1-2 seconds per 100 reviews
- **Total Time (1000 reviews)**: ~10-20 minutes

### Subsequent Runs:
- Much faster (model cached, data processed)
- Can load preprocessed CSV directly

### Tips for Faster Processing:
- Use `sample_size=1000` for testing
- Process full dataset overnight
- Save/load processed data

---

## 🐛 Troubleshooting

### "File not found"
→ Check that `data/amazon_reviews.csv` exists

### "Column not found"
→ Your CSV needs `review_text` (or `reviewText`) and `product_name` (or `itemName`)

### "TensorFlow error"
→ Fixed automatically (environment variables set)

### "Very slow processing"
→ Normal for first run (BERT download). Use smaller sample for testing.

---

## 📝 Output Files

- **`data/Processed_Reviews.csv`**: Processed data with keywords (can reuse this)

---

## 🎓 How Recommendations Work

1. **Review Analysis**: Each product's reviews are analyzed
2. **Keyword Extraction**: BERT finds key phrases describing the product
3. **Product Profile**: All keywords per product are combined
4. **Similarity Matching**: Products with similar keyword profiles are matched
5. **Ranking**: Results sorted by similarity score

---

## ✅ Success Checklist

- [ ] CSV file placed in `data/` folder
- [ ] Required columns present (review_text, product_name)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] App runs without errors
- [ ] Can search and get recommendations

---

**Happy Recommending! 🛒✨**

