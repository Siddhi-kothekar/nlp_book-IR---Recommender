"""
Streamlit App for Amazon Review-Based Recommendation System
"""
import os
import sys
import pandas as pd
import numpy as np
import streamlit as st

# Disable TensorFlow
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from amazon_recommender import (
    load_and_preprocess_data,
    extract_keywords_reviews,
    aggregate_keywords_by_product,
    build_recommendation_system,
    recommend_products
)


@st.cache_data
def load_and_process_data(csv_path, sample_size=None):
    """Load and process data (cached)"""
    df = load_and_preprocess_data(csv_path, sample_size=sample_size)
    df = extract_keywords_reviews(df, top_n=10)
    product_df = aggregate_keywords_by_product(df)
    tfidf, tfidf_matrix, cosine_sim = build_recommendation_system(product_df)
    return product_df, tfidf, tfidf_matrix, cosine_sim


def main():
    st.set_page_config(
        page_title="Amazon Review Recommender",
        page_icon="🛒",
        layout="wide"
    )
    
    st.title("🛒 Amazon Review-Based Product Recommender")
    st.markdown("**Content-Based Recommendations using BERT + TF-IDF**")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        csv_path = st.text_input(
            "Data File Path",
            value="data/amazon_reviews.csv"
        )
        
        use_sample = st.checkbox("Use Sample Data (Faster)", value=True)
        sample_size = 5000 if use_sample else None
        if use_sample:
            sample_size = st.slider(
                "Sample Size",
                min_value=1000,
                max_value=20000,
                value=5000,
                step=500
            )
        
        n_recommendations = st.slider(
            "Number of Recommendations",
            min_value=3,
            max_value=20,
            value=5
        )
    
    # Load data
    if st.button("Load & Process Data") or 'data_loaded' in st.session_state:
        try:
            with st.spinner("Loading and processing data (this may take a few minutes)..."):
                if 'data_loaded' not in st.session_state:
                    product_df, tfidf, tfidf_matrix, cosine_sim = load_and_process_data(
                        csv_path, 
                        sample_size=sample_size
                    )
                    st.session_state['product_df'] = product_df
                    st.session_state['tfidf'] = tfidf
                    st.session_state['cosine_sim'] = cosine_sim
                    st.session_state['data_loaded'] = True
                    st.success("✅ Data loaded and processed successfully!")
                else:
                    product_df = st.session_state['product_df']
                    cosine_sim = st.session_state['cosine_sim']
            
            # Product search
            st.header("🔍 Find Similar Products")
            
            # Product selector
            product_list = product_df['product_name'].tolist()
            selected_product = st.selectbox(
                "Select a Product:",
                options=[""] + sorted(product_list),
                format_func=lambda x: "Choose a product..." if x == "" else x
            )
            
            # Manual input
            manual_input = st.text_input("Or enter product name manually:")
            input_product = manual_input if manual_input else selected_product
            
            if st.button("Get Recommendations", type="primary") and input_product:
                with st.spinner("Finding similar products..."):
                    recommendations = recommend_products(
                        product_df,
                        cosine_sim,
                        input_product,
                        n=n_recommendations
                    )
                
                if recommendations is None or len(recommendations) == 0:
                    st.warning(f"Product '{input_product}' not found or no recommendations available.")
                else:
                    st.success(f"Found {len(recommendations)} similar products!")
                    
                    # Display recommendations
                    for i, (idx, row) in enumerate(recommendations.iterrows(), 1):
                        with st.expander(f"**{i}. {row['product_name']}** (Similarity: {row['similarity_score']:.4f})"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if pd.notna(row['product_id']):
                                    st.write(f"**Product ID:** {row['product_id']}")
                                st.write(f"**Similarity Score:** {row['similarity_score']:.4f}")
                            
                            with col2:
                                st.metric("Similarity", f"{row['similarity_score']*100:.1f}%")
                            
                            if pd.notna(row['review_text']) and len(str(row['review_text'])) > 0:
                                st.write(f"**Sample Review:** {row['review_text'][:200]}...")
            
            # Statistics
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 📊 Statistics")
            st.sidebar.metric("Total Products", len(product_df))
            
        except FileNotFoundError:
            st.error(f"❌ File not found: {csv_path}\n\nPlease ensure your data file exists.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Make sure your CSV has columns: product_name (or itemName/title) and review_text (or reviewText/review)")
    else:
        st.info("👆 Click 'Load & Process Data' to start")
        st.markdown("""
        ### 📋 How to Use
        
        1. **Ensure your data file** is at `data/amazon_reviews.csv`
        2. **Click "Load & Process Data"** - this will:
           - Load and clean reviews
           - Extract keywords using BERT
           - Build similarity matrix
        3. **Select or type a product name**
        4. **Get recommendations!**
        
        ### 📝 Required CSV Columns
        
        Your CSV should have:
        - `product_name` (or `itemName` / `title`)
        - `review_text` (or `reviewText` / `review`)
        - `product_id` (or `asin`) - optional
        
        ### ⚡ Note
        
        First run will take time due to BERT model download and processing.
        Subsequent runs are much faster!
        """)


if __name__ == "__main__":
    main()

