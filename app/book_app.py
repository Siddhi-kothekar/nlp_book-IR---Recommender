"""
Streamlit Application for Book Recommendation System
Content-Based Filtering using BERT Embeddings and TF-IDF
"""
import os
import sys
import json

# Disable TensorFlow in Transformers
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import streamlit as st
import yaml
from src.book_recommender import BookRecommender
from src.book_vectorizer import BookVectorizer


@st.cache_resource
def load_book_resources(cfg_path: str):
    """Load book data and models"""
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    artifacts_dir = cfg.get("artifacts_dir", "artifacts")
    books_path = cfg.get("books_data_path", "data/keywords.csv")
    vectors_path = os.path.join(artifacts_dir, "book_vectors.joblib")
    metadata_path = os.path.join(artifacts_dir, "book_metadata.json")
    
    # Load book data
    if not os.path.exists(books_path):
        st.error(f"Book data not found at {books_path}. Please run the preprocessing pipeline first.")
        st.stop()
    
    books_df = pd.read_csv(books_path)
    
    # Load or create vectorizer
    vectorizer = BookVectorizer(
        min_df=cfg.get("min_df", 3),
        max_df=cfg.get("max_df", 0.6)
    )
    
    if os.path.exists(vectors_path):
        try:
            vectorizer.load_vectors(vectors_path)
        except Exception as e:
            st.warning(f"Could not load vectors: {e}. Recomputing...")
            vectorizer.fit_transform(books_df["keywords"])
            vectorizer.compute_similarity()
            vectorizer.save_vectors(vectors_path)
    else:
        vectorizer.fit_transform(books_df["keywords"])
        vectorizer.compute_similarity()
        vectorizer.save_vectors(vectors_path)
    
    # Load metadata
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        # Create metadata
        metadata = {
            "names": books_df["Name"].tolist(),
            "ids": books_df.get("Id", books_df.index).tolist(),
            "authors": books_df.get("Authors", [""] * len(books_df)).tolist(),
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)
    
    # Create recommender
    book_names = pd.Series(metadata["names"])
    recommender = BookRecommender(vectorizer.similarity_matrix, book_names)
    
    return cfg, books_df, vectorizer, recommender, metadata


def main():
    st.set_page_config(
        page_title="Book Recommendation System",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Content-Based Book Recommendation System")
    st.markdown("**NLP-Based Recommendations Using BERT Embeddings and TF-IDF**")
    
    # Load resources
    try:
        cfg, books_df, vectorizer, recommender, metadata = load_book_resources("config_books.yaml")
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        st.stop()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        n_recommendations = st.slider(
            "Number of Recommendations", 
            min_value=3, 
            max_value=20, 
            value=int(cfg.get("n_recommendations", 5))
        )
        show_keywords = st.checkbox("Show Recommendation Reasons", value=True)
        top_keywords = st.slider(
            "Top Keywords to Show", 
            min_value=3, 
            max_value=10, 
            value=5
        ) if show_keywords else 5
    
    # Main content area
    tab1, tab2 = st.tabs(["🔍 Book Recommendations", "📖 Information Retrieval"])
    
    with tab1:
        st.header("Find Similar Books")
        st.markdown("Enter a book name to get recommendations based on content similarity.")
        
        # Book selection
        book_list = metadata["names"]
        selected_book = st.selectbox(
            "Select a book:",
            options=[""] + sorted(book_list),
            format_func=lambda x: "Type or select a book..." if x == "" else x
        )
        
        # Also allow manual input
        manual_input = st.text_input("Or enter book name manually:")
        input_book = manual_input if manual_input else selected_book
        
        if st.button("Get Recommendations", type="primary") and input_book:
            with st.spinner("Finding similar books..."):
                recommendations = recommender.recommend_similar_books(
                    input_book, 
                    n=n_recommendations
                )
            
            if not recommendations:
                st.warning(f"Book '{input_book}' not found in the database. Try another book.")
            else:
                st.success(f"Found {len(recommendations)} similar books!")
                
                # Display recommendations
                for i, (book_name, score) in enumerate(recommendations, 1):
                    with st.expander(f"**{i}. {book_name}** (Similarity: {score:.4f})"):
                        # Get book details
                        book_idx = books_df[books_df["Name"] == book_name].index
                        if len(book_idx) > 0:
                            book_data = books_df.iloc[book_idx[0]]
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if "Authors" in book_data:
                                    st.write(f"**Author:** {book_data['Authors'].replace('_', ' ')}")
                                if "Language" in book_data:
                                    st.write(f"**Language:** {book_data['Language']}")
                            
                            with col2:
                                if "Id" in book_data:
                                    st.write(f"**Book ID:** {book_data['Id']}")
                            
                            # Show recommendation reasons
                            if show_keywords:
                                reasons = recommender.get_recommendation_reasons(
                                    input_book,
                                    [(book_name, score)],
                                    vectorizer,
                                    top_keywords
                                )
                                if book_name in reasons and reasons[book_name]:
                                    st.markdown("**Common Keywords:**")
                                    st.write(", ".join(reasons[book_name][:top_keywords]))
    
    with tab2:
        st.header("Information Retrieval Search")
        st.markdown("Search for books using keywords, themes, or descriptions.")
        
        query = st.text_input(
            "Enter search query:",
            placeholder="e.g., 'programming python computer science' or 'mystery thriller detective'"
        )
        
        top_k = st.slider(
            "Number of Results",
            min_value=5,
            max_value=50,
            value=int(cfg.get("ir_top_k", 10))
        )
        
        if st.button("Search Books", type="primary") and query:
            with st.spinner(f"Searching for books matching '{query}'..."):
                results = recommender.search_books(query, vectorizer, top_k=top_k)
            
            if not results:
                st.info("No books found matching your query. Try different keywords.")
            else:
                st.success(f"Found {len(results)} books matching your query!")
                
                # Display search results
                for i, (book_name, score) in enumerate(results, 1):
                    with st.expander(f"**{i}. {book_name}** (Relevance: {score:.4f})"):
                        book_idx = books_df[books_df["Name"] == book_name].index
                        if len(book_idx) > 0:
                            book_data = books_df.iloc[book_idx[0]]
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if "Authors" in book_data:
                                    st.write(f"**Author:** {book_data['Authors'].replace('_', ' ')}")
                            
                            with col2:
                                if "Id" in book_data:
                                    st.write(f"**Book ID:** {book_data['Id']}")
                            
                            # Show top keywords for this book
                            book_index = recommender.find_book_index(book_name)
                            if book_index is not None:
                                keywords = vectorizer.get_top_keywords(book_index, top_keywords)
                                if keywords:
                                    st.markdown("**Top Keywords:**")
                                    keyword_text = ", ".join([f"{kw} ({score:.3f})" for kw, score in keywords])
                                    st.write(keyword_text)
    
    # Footer with information
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Statistics")
    st.sidebar.metric("Total Books", len(books_df))
    st.sidebar.metric("Vocabulary Size", len(vectorizer.feature_names) if vectorizer.feature_names is not None else 0)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.markdown("""
    This system uses:
    - **KeyBERT** for keyword extraction from descriptions
    - **TF-IDF** for document vectorization
    - **Cosine Similarity** for finding similar books
    """)


if __name__ == "__main__":
    main()

