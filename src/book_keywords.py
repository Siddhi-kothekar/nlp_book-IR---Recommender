"""
Keyword Extraction Module using KeyBERT
Extracts relevant keywords from book descriptions using BERT embeddings
"""
import pandas as pd
from typing import Optional
from keybert import KeyBERT


class BookKeywordExtractor:
    """Extract keywords from book descriptions using KeyBERT"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize KeyBERT model
        
        Args:
            model_name: Sentence transformer model name for KeyBERT
        """
        self.kw_model = KeyBERT(model_name)
    
    def extract_keywords(self, text: str, top_n: int = 10, 
                        ngram_range: tuple = (1, 1)) -> str:
        """
        Extract keywords from text using KeyBERT
        
        Args:
            text: Input text to extract keywords from
            top_n: Number of top keywords to extract
            ngram_range: Range of n-grams (default: unigrams)
        
        Returns:
            Space-separated string of keywords
        """
        if not text or pd.isna(text):
            return ""
        
        try:
            keywords = self.kw_model.extract_keywords(
                str(text),
                keyphrase_ngram_range=ngram_range,
                stop_words="english",
                top_n=top_n
            )
            # Extract just the keyword phrases (first element of each tuple)
            keywords_list = [k[0] for k in keywords if k[1] > 0.1]  # filter low scores
            return " ".join(keywords_list)
        except Exception as e:
            print(f"Error extracting keywords: {e}")
            return ""
    
    def process_books(self, df: pd.DataFrame, 
                     description_col: str = 'Description',
                     bow_col: str = 'bow',
                     top_n: int = 10) -> pd.DataFrame:
        """
        Extract keywords for all books in dataframe
        
        Args:
            df: DataFrame with book descriptions
            description_col: Column name containing descriptions
            bow_col: Column name containing bag of words metadata
            top_n: Number of keywords to extract per book
        
        Returns:
            DataFrame with extracted keywords
        """
        print(f"Extracting keywords from {len(df)} books...")
        
        # Extract keywords from descriptions
        df['keywords'] = df[description_col].apply(
            lambda x: self.extract_keywords(x, top_n=top_n)
        )
        
        # Combine with metadata (bow)
        df['keywords'] = df[[bow_col, 'keywords']].fillna('').apply(
            lambda x: ' '.join(x).strip(), axis=1
        )
        
        # Remove duplicates based on book name
        df = df.drop_duplicates(subset=['Name'], keep='first')
        
        return df

