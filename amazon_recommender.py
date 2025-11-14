"""
Amazon Review-Based Recommendation System
Content-Based Filtering using BERT (KeyBERT) + TF-IDF + Cosine Similarity
"""
import pandas as pd
import numpy as np
import re
import string
import os
import sys
import subprocess
from shutil import which
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Disable TensorFlow
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def clean_text(text):
    """Clean review text by removing URLs, HTML tags, and punctuations"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.lower().strip()


def load_and_preprocess_data(csv_path="data/amazon_reviews.csv", sample_size=None):
    """
    Load and preprocess Amazon reviews data
    
    Args:
        csv_path: Path to CSV file
        sample_size: Number of rows to sample (None = use all)
    
    Returns:
        Preprocessed DataFrame
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Map common column names
    column_map = {
        # Generic/Amazon
        'asin': 'product_id',
        'itemName': 'product_name',
        'title': 'product_name',
        'reviewText': 'review_text',
        'review': 'review_text',
        'overall': 'rating',
        # Books dataset variants
        'name': 'product_name',
        'book_title': 'product_name',
        'Book-Title': 'product_name',
        'Title': 'product_name',
        'authors': 'author',
        'author': 'author',
        'Authors': 'author',
        'publisher': 'publisher',
        'Publisher': 'publisher',
        'genre': 'genre',
        'categories': 'genre',
        'Category': 'genre',
        'Genres': 'genre',
        'description': 'review_text',
        'Description': 'review_text',
        'summary': 'review_text',
        'Average rating': 'rating',
        'average_rating': 'rating',
        'rating': 'rating',
        'Rating': 'rating',
        'ratingDistTotal': 'rating_count',
        'ratings_count': 'rating_count',
        'rating_count': 'rating_count',
        'reviews': 'rating_count',
        'num_ratings': 'rating_count',
    }
    
    # Rename columns if needed
    for old_col, new_col in column_map.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]
    
    # Keep relevant columns
    required_cols = ['product_id', 'product_name', 'review_text']
    available_cols = [col for col in required_cols if col in df.columns]
    
    if not available_cols:
        # Try to find any product and review columns
        product_cols = [c for c in df.columns if 'product' in c.lower() or 'item' in c.lower() or 'asin' in c.lower()]
        review_cols = [c for c in df.columns if 'review' in c.lower() or 'text' in c.lower()]
        
        if product_cols:
            df['product_name'] = df[product_cols[0]]
        if review_cols:
            df['review_text'] = df[review_cols[0]]
        if 'asin' in df.columns:
            df['product_id'] = df['asin']
        
        available_cols = ['product_name', 'review_text']
    
    df = df[available_cols + [c for c in df.columns if c not in available_cols]]

    # Ensure product_name exists (books datasets may not have explicit product field)
    if 'product_name' not in df.columns:
        title_candidates = [
            'title', 'Title', 'name', 'book_title', 'Book-Title', 'bookName', 'book_name',
        ]
        found_title = None
        for cand in title_candidates:
            if cand in df.columns:
                found_title = cand
                break
        if found_title is not None:
            df['product_name'] = df[found_title].astype(str)
        else:
            # fallback to author + description slice
            author_part = df.get('author', pd.Series([""] * len(df))).astype(str)
            desc_part = df.get('review_text', pd.Series([""] * len(df))).astype(str).str.slice(0, 60)
            df['product_name'] = (author_part.str.cat(desc_part, sep=' - ').str.strip()).replace('', np.nan)
            fallback_ids = pd.Series(np.arange(len(df)).astype(str), index=df.index)
            df['product_name'] = df['product_name'].fillna(fallback_ids)

    # Build review_text from available textual fields if missing or empty
    if 'review_text' not in df.columns:
        text_parts = []
        if 'reviewText' in df.columns:
            text_parts.append(df['reviewText'].astype(str))
        if 'summary' in df.columns:
            text_parts.append(df['summary'].astype(str))
        if 'description' in df.columns:
            text_parts.append(df['description'].astype(str))
        if text_parts:
            df['review_text'] = pd.Series([""] * len(df))
            # Concatenate available parts per row
            df['review_text'] = (
                (text_parts[0] if len(text_parts) > 0 else "")
                .str.cat(text_parts[1] if len(text_parts) > 1 else "", sep=" ", na_rep="")
                .str.cat(text_parts[2] if len(text_parts) > 2 else "", sep=" ", na_rep="")
                .str.strip()
            )
    else:
        # If review_text exists but has empties, try to enrich with summary/description/title/etc.
        supplemental_parts = []
        for col in ['summary', 'description', 'Description', 'Title', 'title', 'name', 'book_title', 'Book-Title', 'author', 'publisher', 'genre']:
            if col in df.columns:
                supplemental_parts.append(df[col].astype(str))
        if supplemental_parts:
            supplemental = supplemental_parts[0]
            for p in supplemental_parts[1:]:
                supplemental = supplemental.str.cat(p, sep=" ", na_rep="")
            supplemental = supplemental.str.strip()
            df.loc[df['review_text'].isna() | (df['review_text'].astype(str).str.strip()==""), 'review_text'] = supplemental

    # Ensure product_id exists to avoid downstream KeyError
    if 'product_id' not in df.columns:
        # Fallback: use product_name as a stable identifier
        if 'product_name' in df.columns:
            df['product_id'] = df['product_name'].astype(str)
        else:
            # As a last resort, create a sequential id
            df['product_id'] = np.arange(len(df)).astype(str)
    
    # Drop rows with missing reviews
    df = df.dropna(subset=['review_text'])

    # Normalize optional fields
    for opt_col in ['author', 'publisher', 'genre']:
        if opt_col not in df.columns:
            df[opt_col] = ""
        else:
            df[opt_col] = df[opt_col].fillna("")

    # Coerce rating and rating_count if present
    if 'rating' in df.columns:
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    else:
        df['rating'] = np.nan
    if 'rating_count' in df.columns:
        df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce')
    else:
        df['rating_count'] = np.nan
    print(f"Loaded {len(df)} reviews")
    
    # Sample if requested
    if sample_size and sample_size < len(df):
        print(f"Sampling {sample_size} rows for faster processing...")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    # Clean text
    print("Cleaning review text...")
    df['clean_review'] = df['review_text'].apply(clean_text)
    
    # Remove empty reviews after cleaning
    df = df[df['clean_review'].str.len() > 0]
    
    return df


def extract_keywords_reviews(df, model_name="all-MiniLM-L6-v2", top_n=10):
    """
    Extract keywords from reviews using KeyBERT
    
    Args:
        df: DataFrame with 'clean_review' column
        model_name: KeyBERT model name
        top_n: Number of keywords to extract per review
    
    Returns:
        DataFrame with 'keywords' column added
    """
    print(f"Initializing KeyBERT model: {model_name}...")
    kw_model = KeyBERT(model_name)
    
    def extract_keywords(text):
        """Extract keywords from a single review"""
        if not text or len(text.strip()) < 10:
            return ""
        try:
            keywords = kw_model.extract_keywords(
                text, 
                keyphrase_ngram_range=(1, 2), 
                stop_words='english',
                top_n=top_n
            )
            return " ".join([kw[0] for kw in keywords if kw[1] > 0.1])
        except Exception as e:
            print(f"Error extracting keywords: {e}")
            return ""
    
    print("Extracting keywords from reviews (this may take time)...")
    df['keywords'] = df['clean_review'].apply(extract_keywords)
    
    # Remove rows with no keywords
    df = df[df['keywords'].str.len() > 0]
    print(f"Extracted keywords for {len(df)} reviews")
    
    return df


def aggregate_keywords_by_product(df):
    """
    Aggregate keywords by product (combine all review keywords per product)
    
    Args:
        df: DataFrame with 'product_name' and 'keywords' columns
    
    Returns:
        DataFrame with one row per product and aggregated keywords
    """
    print("Aggregating keywords by product...")
    
    # Group by product and combine keywords
    product_keywords = df.groupby('product_name')['keywords'].apply(
        lambda x: " ".join(x.astype(str))
    ).reset_index()
    
    # Get other product info (only aggregate columns that exist)
    agg_spec = {}
    if 'product_id' in df.columns:
        agg_spec['product_id'] = 'first'
    if 'review_text' in df.columns:
        agg_spec['review_text'] = lambda x: x.iloc[0] if len(x) > 0 else ""
    if 'author' in df.columns:
        agg_spec['author'] = 'first'
    if 'publisher' in df.columns:
        agg_spec['publisher'] = 'first'
    if 'genre' in df.columns:
        agg_spec['genre'] = 'first'
    if 'rating' in df.columns:
        agg_spec['rating'] = 'max'
    if 'rating_count' in df.columns:
        agg_spec['rating_count'] = 'max'

    if agg_spec:
        product_info = df.groupby('product_name').agg(agg_spec).reset_index()
    else:
        product_info = df[['product_name']].drop_duplicates().copy()
    
    # Merge
    product_df = product_keywords.merge(product_info, on='product_name', how='left')
    
    print(f"Aggregated {len(product_df)} products")
    return product_df


def build_recommendation_system(product_df, min_df=2, max_df=0.7):
    """
    Build TF-IDF vectors and cosine similarity matrix
    
    Args:
        product_df: DataFrame with product keywords
        min_df: Minimum document frequency
        max_df: Maximum document frequency
    
    Returns:
        tuple: (tfidf_vectorizer, tfidf_matrix, cosine_similarity_matrix)
    """
    print("Building TF-IDF vectors...")
    tfidf = TfidfVectorizer(
        stop_words='english',
        max_df=max_df,
        min_df=min_df
    )
    tfidf_matrix = tfidf.fit_transform(product_df['keywords'])
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Vocabulary size: {len(tfidf.get_feature_names_out())}")
    
    print("Computing cosine similarity matrix...")
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return tfidf, tfidf_matrix, cosine_sim


def recommend_products(product_df, cosine_sim, product_name, n=5, genre_filter=None, sort_by_rating=True):
    """
    Recommend similar products based on cosine similarity
    
    Args:
        product_df: DataFrame with products
        cosine_sim: Cosine similarity matrix
        product_name: Name of product to find recommendations for
        n: Number of recommendations
    
    Returns:
        DataFrame with recommended products
    """
    # Optionally filter by genre first
    filtered_df = product_df
    if genre_filter:
        mask = filtered_df.get('genre', pd.Series([""] * len(filtered_df))).astype(str).str.contains(str(genre_filter), case=False, na=False)
        if mask.any():
            filtered_df = filtered_df[mask].reset_index(drop=True)
        else:
            # No genre matches; fall back to full set
            filtered_df = product_df

    # Create index mapping
    indices = pd.Series(product_df.index, index=product_df['product_name']).drop_duplicates()
    
    if product_name not in indices:
        # Try case-insensitive search
        product_names_lower = product_df['product_name'].str.lower()
        matches = product_names_lower[product_names_lower == product_name.lower()]
        if len(matches) == 0:
            return None
        product_name = product_df.loc[matches.index[0], 'product_name']
    
    idx = indices[product_name]
    
    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort by similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    
    # Get product indices
    product_indices = [i[0] for i in sim_scores]
    
    # Return recommendations
    cols = ['product_name', 'product_id', 'review_text']
    for extra in ['author', 'publisher', 'genre', 'rating', 'rating_count']:
        if extra in product_df.columns and extra not in cols:
            cols.append(extra)
    recommendations = product_df[cols].iloc[product_indices].copy()
    recommendations['similarity_score'] = [i[1] for i in sim_scores]

    # If a genre filter is provided and sorting by rating is desired, prioritize higher ratings/counts
    if genre_filter and sort_by_rating:
        if 'rating' in recommendations.columns or 'rating_count' in recommendations.columns:
            recommendations = recommendations.sort_values(
                by=[c for c in ['rating', 'rating_count', 'similarity_score'] if c in recommendations.columns],
                ascending=[False, False, False]
            )
    
    return recommendations


def main():
    """Main workflow"""
    print("=" * 60)
    print("Amazon Review-Based Recommendation System")
    print("Content-Based Filtering using BERT + TF-IDF")
    print("=" * 60)
    
    # Step 1: Load and preprocess
    # Default to provided books dataset
    csv_path = "data/book2000k-3000k.csv"
    
    # Use a subset for faster processing on large files
    SAMPLE_SIZE = 1000  # change to None to use full dataset
    
    df = load_and_preprocess_data(csv_path, sample_size=SAMPLE_SIZE)
    
    # Step 2: Extract keywords
    df = extract_keywords_reviews(df, top_n=10)
    
    # Step 3: Aggregate by product
    product_df = aggregate_keywords_by_product(df)
    
    # Step 4: Build recommendation system
    tfidf, tfidf_matrix, cosine_sim = build_recommendation_system(product_df)
    
    # Step 5: Save processed data
    output_path = "data/Processed_Books.csv"
    product_df.to_csv(output_path, index=False)
    print(f"\nSaved processed data to {output_path}")

    # Also save keywords in the format expected by the Streamlit app (data/keywords.csv)
    keywords_out = "data/keywords.csv"
    keywords_df = pd.DataFrame({
        'Name': product_df['product_name'],
        'keywords': product_df['keywords'],
        'Id': product_df.get('product_id', pd.Series(range(len(product_df)))).astype(str),
        'Authors': product_df.get('author', pd.Series([""] * len(product_df))).astype(str),
    })
    keywords_df.to_csv(keywords_out, index=False)
    print(f"Saved keywords for app to {keywords_out}")
    
    # Step 6: Example recommendation
    print("\n" + "=" * 60)
    print("Example Recommendations")
    print("=" * 60)
    
    # Example 1: similar to first product
    if len(product_df) > 0:
        example_product = product_df.iloc[0]['product_name']
        print(f"\nFinding products similar to: '{example_product}'")
        recommendations = recommend_products(product_df, cosine_sim, example_product, n=5)
        if recommendations is not None and len(recommendations) > 0:
            print(f"\nTop {len(recommendations)} Recommendations:")
            print("-" * 60)
            for i, (idx, row) in enumerate(recommendations.iterrows(), 1):
                print(f"{i}. {row['product_name']}")
                if 'author' in row:
                    print(f"   Author: {row['author']}")
                if 'genre' in row:
                    print(f"   Genre: {row['genre']}")
                if 'rating' in row and not pd.isna(row['rating']):
                    print(f"   Rating: {row['rating']}")
                if 'rating_count' in row and not pd.isna(row['rating_count']):
                    print(f"   Ratings Count: {int(row['rating_count'])}")
                print(f"   Similarity: {row['similarity_score']:.4f}")
                print(f"   Description: {str(row['review_text'])[:100]}...")
                print()
        else:
            print("No recommendations found.")

    # Example 2: genre-based (e.g., Thriller)
    print("\n" + "-" * 60)
    print("Top Thriller recommendations (by rating & similarity):")
    print("-" * 60)
    try:
        # Use the first product that matches genre as anchor if available, else first product overall
        anchor_idx = 0
        if 'genre' in product_df.columns:
            thriller_mask = product_df['genre'].astype(str).str.contains('thriller', case=False, na=False)
            if thriller_mask.any():
                anchor_idx = product_df[thriller_mask].index[0]
        anchor_name = product_df.iloc[anchor_idx]['product_name']
        thriller_recs = recommend_products(product_df, cosine_sim, anchor_name, n=10, genre_filter='thriller', sort_by_rating=True)
        if thriller_recs is not None and len(thriller_recs) > 0:
            for i, (idx, row) in enumerate(thriller_recs.iterrows(), 1):
                print(f"{i}. {row['product_name']}")
                if 'author' in row:
                    print(f"   Author: {row['author']}")
                if 'genre' in row:
                    print(f"   Genre: {row['genre']}")
                if 'rating' in row and not pd.isna(row['rating']):
                    print(f"   Rating: {row['rating']}")
                if 'rating_count' in row and not pd.isna(row['rating_count']):
                    print(f"   Ratings Count: {int(row['rating_count'])}")
                print(f"   Similarity: {row['similarity_score']:.4f}")
                print(f"   Description: {str(row['review_text'])[:100]}...")
                print()
        else:
            print("No thriller recommendations found.")
    except Exception as e:
        print(f"Genre-based recommendation skipped due to error: {e}")
    
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print("\nTo get recommendations for a product, use:")
    print("  recommendations = recommend_products(product_df, cosine_sim, 'Your Product Name', n=5)")
    print("\nOr use the Streamlit app:")
    print("  streamlit run app/amazon_recommender_app.py")

    # Auto-launch Streamlit app in browser so running this file shows UI
    try:
        app_path = os.path.join("app", "book_app.py")
        # Prefer current interpreter; fallback to local venv if streamlit is missing here
        def _launch(python_exe):
            print("\nStarting Streamlit app at http://localhost:8501 ...")
            return subprocess.Popen([python_exe, "-m", "streamlit", "run", app_path, "--server.port", "8501"])  # noqa: S603

        # Check if streamlit available in current python
        if which("streamlit") is not None:
            _ = _launch(sys.executable)
        else:
            # Try project venv
            venv_python = os.path.join(".venv", "Scripts", "python.exe") if os.name == "nt" else os.path.join(".venv", "bin", "python")
            if os.path.exists(venv_python):
                _ = _launch(venv_python)
            else:
                print("Streamlit not found. Please run: .\\.venv\\Scripts\\activate && streamlit run app/book_app.py")
    except Exception as _e:
        print(f"Could not launch Streamlit automatically: {_e}")


if __name__ == "__main__":
    main()

