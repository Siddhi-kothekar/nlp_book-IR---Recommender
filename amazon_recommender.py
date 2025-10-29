"""
Amazon Review-Based Recommendation System
Content-Based Filtering using BERT (KeyBERT) + TF-IDF + Cosine Similarity
"""
import pandas as pd
import numpy as np
import re
import string
import os
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
        'asin': 'product_id',
        'itemName': 'product_name',
        'title': 'product_name',
        'reviewText': 'review_text',
        'review': 'review_text',
        'rating': 'rating',
        'overall': 'rating'
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
        # If review_text exists but has empties, try to enrich with summary/description
        if 'summary' in df.columns or 'description' in df.columns:
            supplemental = (
                df.get('summary', pd.Series([""] * len(df))).astype(str)
                .str.cat(df.get('description', pd.Series([""] * len(df))).astype(str), sep=" ", na_rep="")
            ).str.strip()
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


def recommend_products(product_df, cosine_sim, product_name, n=5):
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
    recommendations = product_df[['product_name', 'product_id', 'review_text']].iloc[product_indices].copy()
    recommendations['similarity_score'] = [i[1] for i in sim_scores]
    
    return recommendations


def main():
    """Main workflow"""
    print("=" * 60)
    print("Amazon Review-Based Recommendation System")
    print("Content-Based Filtering using BERT + TF-IDF")
    print("=" * 60)
    
    # Step 1: Load and preprocess
    csv_path = "data/amazon_reviews.csv"
    
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
    output_path = "data/Processed_Reviews.csv"
    product_df.to_csv(output_path, index=False)
    print(f"\nSaved processed data to {output_path}")
    
    # Step 6: Example recommendation
    print("\n" + "=" * 60)
    print("Example Recommendations")
    print("=" * 60)
    
    # Try to recommend first product as example
    if len(product_df) > 0:
        example_product = product_df.iloc[0]['product_name']
        print(f"\nFinding products similar to: '{example_product}'")
        
        recommendations = recommend_products(product_df, cosine_sim, example_product, n=5)
        
        if recommendations is not None and len(recommendations) > 0:
            print(f"\nTop {len(recommendations)} Recommendations:")
            print("-" * 60)
            for i, (idx, row) in enumerate(recommendations.iterrows(), 1):
                print(f"{i}. {row['product_name']}")
                print(f"   Similarity: {row['similarity_score']:.4f}")
                print(f"   Sample Review: {row['review_text'][:100]}...")
                print()
        else:
            print("No recommendations found.")
    
    print("\n" + "=" * 60)
    print("✅ Processing Complete!")
    print("=" * 60)
    print("\nTo get recommendations for a product, use:")
    print("  recommendations = recommend_products(product_df, cosine_sim, 'Your Product Name', n=5)")
    print("\nOr use the Streamlit app:")
    print("  streamlit run app/amazon_recommender_app.py")


if __name__ == "__main__":
    main()

