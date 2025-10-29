"""
Example Script: Book Recommendation System Usage
Demonstrates how to use the book recommendation system
"""
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from src.book_processor import BookProcessor
from src.book_keywords import BookKeywordExtractor
from src.book_vectorizer import BookVectorizer
from src.book_recommender import BookRecommender


def example_usage():
    """Complete example of using the book recommendation system"""
    
    print("=" * 60)
    print("Book Recommendation System - Example Usage")
    print("=" * 60)
    
    # Step 1: Load and preprocess data
    print("\n[Step 1] Loading and preprocessing books...")
    books_data = pd.read_csv(
        "data/goodreads_book.csv",  # Replace with your data path
        usecols=['Id', 'Name', 'Authors', 'ISBN', 'PublishYear', 
                 'Publisher', 'Language', 'Description']
    )
    
    processor = BookProcessor(min_description_word_count=3)
    books_data = processor.process_books(books_data)
    print(f"✓ Processed {len(books_data)} books")
    
    # Step 2: Extract keywords
    print("\n[Step 2] Extracting keywords using KeyBERT...")
    extractor = BookKeywordExtractor(model_name="all-MiniLM-L6-v2")
    books_data = extractor.process_books(
        books_data,
        description_col="Description",
        bow_col="bow",
        top_n=10
    )
    print(f"✓ Extracted keywords for {len(books_data)} books")
    
    # Step 3: Vectorize books
    print("\n[Step 3] Vectorizing books with TF-IDF...")
    vectorizer = BookVectorizer(min_df=3, max_df=0.6)
    vectorizer.fit_transform(books_data["keywords"])
    similarity_matrix = vectorizer.compute_similarity()
    print(f"✓ Created similarity matrix: {similarity_matrix.shape}")
    
    # Step 4: Create recommender
    print("\n[Step 4] Creating recommender system...")
    recommender = BookRecommender(
        similarity_matrix,
        books_data["Name"]
    )
    print("✓ Recommender ready!")
    
    # Step 5: Get recommendations
    print("\n[Step 5] Getting recommendations...")
    input_book = "the practice of programming (addison-wesley professional computing series)"
    recommendations = recommender.recommend_similar_books(input_book, n=5)
    
    print(f"\n📚 Books similar to: '{input_book}'")
    print("-" * 60)
    for i, (book_name, score) in enumerate(recommendations, 1):
        print(f"{i}. {book_name}")
        print(f"   Similarity Score: {score:.4f}")
        
        # Get recommendation reasons
        reasons = recommender.get_recommendation_reasons(
            input_book,
            [(book_name, score)],
            vectorizer,
            top_keywords=5
        )
        if book_name in reasons and reasons[book_name]:
            print(f"   Common Keywords: {', '.join(reasons[book_name][:5])}")
        print()
    
    # Step 6: Information Retrieval
    print("\n[Step 6] Information Retrieval Search...")
    query = "programming python computer science"
    results = recommender.search_books(query, vectorizer, top_k=5)
    
    print(f"\n🔍 Search Results for: '{query}'")
    print("-" * 60)
    for i, (book_name, score) in enumerate(results, 1):
        print(f"{i}. {book_name}")
        print(f"   Relevance Score: {score:.4f}")
        
        # Show top keywords
        book_idx = recommender.find_book_index(book_name)
        if book_idx is not None:
            keywords = vectorizer.get_top_keywords(book_idx, top_n=5)
            if keywords:
                keyword_text = ", ".join([f"{kw}" for kw, _ in keywords])
                print(f"   Top Keywords: {keyword_text}")
        print()
    
    print("=" * 60)
    print("✅ Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    example_usage()

