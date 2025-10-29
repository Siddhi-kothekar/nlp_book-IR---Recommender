# Quick Start - Amazon Reviews System

## 🎯 3 Simple Steps

### Step 1: Install
```bash
pip install -r requirements.txt
```

### Step 2: Add Your Data
Place `amazon_reviews.csv` in `data/` folder

### Step 3: Run
```bash
streamlit run app/streamlit_app.py
```

**That's it!** Browser opens automatically.

---

## 📝 What the App Does

1. **Loads** your Amazon reviews
2. **Builds** embeddings (first time only, takes a few minutes)
3. **Enables** you to:
   - Search reviews by keywords
   - Get product recommendations
   - Filter by sentiment and ratings

---

## 🖥️ Windows Users

Double-click: **`run_amazon_reviews.bat`**

This sets everything up automatically!

---

## ✅ Success Indicators

You'll know it's working when:
- ✅ Browser opens with Streamlit app
- ✅ You can type in the search box
- ✅ You see "Search & Recommend" button

---

## 🔧 If Something Goes Wrong

1. **"No module named..."** 
   → Run `pip install -r requirements.txt`

2. **"File not found: amazon_reviews.csv"**
   → Check that your file is in `data/` folder

3. **TensorFlow errors**
   → Fixed! Environment variables are set automatically now.

---

## 📊 What You Need

**Required CSV columns:**
- Review text (column name: `reviewText` or `review`)
- Product ID (column name: `asin` or `itemName`)

**Optional:**
- Ratings (`rating` or `overall`)
- Product names (`itemName` or `title`)

---

**Ready to go!** 🚀

