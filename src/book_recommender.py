"""
Book Recommendation Module
Provides content-based recommendations using cosine similarity
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from src.book_vectorizer import BookVectorizer


class BookRecommender:
    """Content-based book recommendation system"""
    
    def __init__(self, similarity_matrix: np.ndarray, book_names: pd.Series):
        """
        Initialize recommender with similarity matrix
        
        Args:
            similarity_matrix: NxN cosine similarity matrix
            book_names: Series of book names (indexed by matrix rows)
        """
        self.similarity_matrix = similarity_matrix
        self.book_names = book_names.reset_index(drop=True)
        self.book_to_index = {name: idx for idx, name in enumerate(self.book_names)}
    
    def find_book_index(self, book_name: str) -> Optional[int]:
        """
        Find index of a book by name (case-insensitive)
        
        Args:
            book_name: Name of the book to find
        
        Returns:
            Index of the book or None if not found
        """
        book_name_lower = str(book_name).lower().strip()
        
        # Exact match
        if book_name_lower in self.book_to_index:
            return self.book_to_index[book_name_lower]
        
        # Fuzzy match (partial)
        for idx, name in enumerate(self.book_names):
            if book_name_lower in name.lower() or name.lower() in book_name_lower:
                return idx
        
        return None
    
    def recommend_similar_books(self, book_name: str, n: int = 5, 
                                exclude_input: bool = True) -> List[Tuple[str, float]]:
        """
        Recommend books similar to the given book
        
        Args:
            book_name: Name of the input book
            n: Number of recommendations
            exclude_input: Whether to exclude the input book from results
        
        Returns:
            List of (book_name, similarity_score) tuples
        """
        input_idx = self.find_book_index(book_name)
        
        if input_idx is None:
            return []
        
        # Get similarity scores for this book
        similarities = self.similarity_matrix[input_idx]
        
        # Get top N similar books
        top_indices = np.argsort(similarities)[::-1]
        
        if exclude_input:
            # Remove the input book itself (should have similarity = 1.0)
            top_indices = top_indices[top_indices != input_idx]
        
        recommendations = []
        for idx in top_indices[:n]:
            if similarities[idx] > 0:
                recommendations.append((
                    self.book_names.iloc[idx],
                    float(similarities[idx])
                ))
        
        return recommendations
    
    def get_recommendation_reasons(self, book_name: str, 
                                   recommendations: List[Tuple[str, float]],
                                   vectorizer: BookVectorizer,
                                   top_keywords: int = 5) -> Dict[str, List[str]]:
        """
        Get keywords explaining why books were recommended
        
        Args:
            book_name: Input book name
            recommendations: List of recommended books from recommend_similar_books
            vectorizer: BookVectorizer instance
            top_keywords: Number of top keywords to show
        
        Returns:
            Dictionary mapping book names to lists of keywords
        """
        input_idx = self.find_book_index(book_name)
        if input_idx is None:
            return {}
        
        # Get input book's keywords
        input_keywords = {kw: score for kw, score in vectorizer.get_top_keywords(input_idx, top_keywords)}
        
        reasons = {}
        for rec_book, score in recommendations:
            rec_idx = self.find_book_index(rec_book)
            if rec_idx is None:
                continue
            
            # Get recommended book's keywords
            rec_keywords = {kw: score for kw, score in vectorizer.get_top_keywords(rec_idx, top_keywords)}
            
            # Find common keywords (these explain the similarity)
            common = set(input_keywords.keys()) & set(rec_keywords.keys())
            reasons[rec_book] = sorted(list(common), key=lambda x: input_keywords.get(x, 0), reverse=True)
        
        return reasons
    
    def search_books(self, query: str, vectorizer: BookVectorizer, 
                     top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search books using query keywords (information retrieval)
        
        Args:
            query: Search query string
            vectorizer: BookVectorizer instance
            top_k: Number of results to return
        
        Returns:
            List of (book_name, relevance_score) tuples
        """
        if vectorizer.tfidf_matrix is None:
            raise ValueError("Vectorizer must be fitted first")
        
        # Transform query to TF-IDF vector
        query_vector = vectorizer.tfidf.transform([query])
        
        # Compute similarity with all books
        similarities = cosine_similarity(query_vector, vectorizer.tfidf_matrix)[0]
        
        # Get top K results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append((
                    self.book_names.iloc[idx],
                    float(similarities[idx])
                ))
        
        return results

