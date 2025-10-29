"""
Book Vectorization Module
Creates TF-IDF vectors from book keywords and computes cosine similarity
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, Optional, List


class BookVectorizer:
    """Vectorize books using TF-IDF and compute similarity"""
    
    def __init__(self, min_df: int = 3, max_df: float = 0.6, 
                 stop_words: str = "english"):
        """
        Initialize TF-IDF vectorizer
        
        Args:
            min_df: Minimum document frequency for a term
            max_df: Maximum document frequency (ratio)
            stop_words: Stop words removal strategy
        """
        self.tfidf = TfidfVectorizer(
            analyzer='word',
            min_df=min_df,
            max_df=max_df,
            stop_words=stop_words,
            encoding='utf-8',
            token_pattern=r"(?u)\S\S+"  # Keep tokens with underscores
        )
        self.tfidf_matrix: Optional[np.ndarray] = None
        self.similarity_matrix: Optional[np.ndarray] = None
        self.book_names: Optional[pd.Series] = None
        self.feature_names: Optional[np.ndarray] = None
    
    def fit_transform(self, keywords: pd.Series) -> np.ndarray:
        """
        Fit TF-IDF vectorizer and transform keywords to vectors
        
        Args:
            keywords: Series of keyword strings
        
        Returns:
            TF-IDF matrix (n_books x n_features)
        """
        print(f"Vectorizing {len(keywords)} books...")
        self.tfidf_matrix = self.tfidf.fit_transform(keywords)
        self.feature_names = self.tfidf.get_feature_names_out()
        print(f"Vocabulary size: {len(self.feature_names)}")
        return self.tfidf_matrix
    
    def compute_similarity(self) -> np.ndarray:
        """
        Compute cosine similarity matrix between all books
        
        Returns:
            Similarity matrix (n_books x n_books)
        """
        if self.tfidf_matrix is None:
            raise ValueError("Must fit_transform first before computing similarity")
        
        print("Computing cosine similarity matrix...")
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        return self.similarity_matrix
    
    def get_top_keywords(self, book_index: int, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Get top keywords for a specific book
        
        Args:
            book_index: Index of the book
            top_n: Number of top keywords to return
        
        Returns:
            List of (keyword, tfidf_score) tuples
        """
        if self.tfidf_matrix is None:
            raise ValueError("Must fit_transform first")
        
        # Get TF-IDF scores for this book
        scores = self.tfidf_matrix[book_index].toarray()[0]
        
        # Get indices of top scores
        top_indices = np.argsort(scores)[-top_n:][::-1]
        
        # Return keywords and scores
        top_keywords = [
            (self.feature_names[idx], float(scores[idx])) 
            for idx in top_indices if scores[idx] > 0
        ]
        
        return top_keywords
    
    def save_vectors(self, path: str) -> None:
        """Save TF-IDF matrix to disk"""
        if self.tfidf_matrix is None:
            raise ValueError("No vectors to save")
        
        try:
            import joblib
        except ImportError:
            print("joblib not installed. Cannot save vectors.")
            return
        
        joblib.dump({
            'matrix': self.tfidf_matrix,
            'similarity': self.similarity_matrix,
            'feature_names': self.feature_names
        }, path)
        print(f"Saved vectors to {path}")
    
    def load_vectors(self, path: str) -> None:
        """Load TF-IDF matrix from disk"""
        try:
            import joblib
        except ImportError:
            raise ImportError("joblib not installed. Cannot load vectors.")
        
        data = joblib.load(path)
        self.tfidf_matrix = data['matrix']
        self.similarity_matrix = data.get('similarity')
        self.feature_names = data['feature_names']
        print(f"Loaded vectors from {path}")

