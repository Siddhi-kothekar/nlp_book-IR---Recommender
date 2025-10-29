"""
Book Data Preprocessing Module
Implements text cleaning and preprocessing for book descriptions
"""
import re
import string
import numpy as np
import pandas as pd
from typing import Optional
from nltk.classify.textcat import TextCat


class BookProcessor:
    """Processes book data for recommendation system"""
    
    def __init__(self, min_description_word_count: int = 3):
        self.min_description_word_count = min_description_word_count
        self.tc = TextCat()
        
    def remove_urls(self, text: str) -> str:
        """Remove URLs from text"""
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return re.sub(url_pattern, '', str(text))
    
    def clean_html_tags(self, text: str) -> str:
        """Remove HTML tags from text"""
        html_pattern = re.compile('<[^>]*>')
        return re.sub(html_pattern, '', str(text))
    
    def remove_punctuations(self, text: str) -> str:
        """Remove punctuations from text"""
        return str(text).translate(str.maketrans('', '', string.punctuation))
    
    def get_book_series_info(self, text: str) -> Optional[str]:
        """Extract book series information from book name"""
        series_pattern = r"(?:[;]\s*|\(\s*)([^\(;]*\s*#\s*\d+(?:\.?\d+|\\&\d+|-?\d*))"
        series_info = re.findall(series_pattern, text)
        if series_info:
            return " ".join([i.replace(" ", "_") for i in series_info])
        return None
    
    def remove_series_info(self, text: str) -> str:
        """Remove series information from book name"""
        series_remove_pattern = re.compile(
            r"(?:[\(]\s*[^\(;]*\s*#\s*\d+(?:\.?\d+|\\&\d+|-?\d*)(?:;|\))|\s*[^\(;]*\s*#\s*\d+(?:\.?\d+|\\&\d+|-?\d*)\))"
        )
        return re.sub(series_remove_pattern, '', str(text)).strip()
    
    def detect_language(self, text: str) -> str:
        """Detect language from text"""
        text_clean = " ".join(str(text).split()[:5])
        if text_clean.isnumeric():
            return 'eng'
        try:
            return self.tc.guess_language(text_clean).strip()
        except:
            return 'eng'
    
    def clean_descriptions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean book descriptions"""
        # Remove URLs and HTML tags
        df['Description'] = df['Description'].apply(self.remove_urls)
        df['Description'] = df['Description'].apply(self.clean_html_tags)
        df['Description'] = df['Description'].apply(self.remove_punctuations)
        
        # Convert to lowercase and strip
        text_cols = ['Name', 'Authors', 'Publisher', 'Description']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().str.strip()
        
        # Fill missing Publisher
        if 'Publisher' in df.columns:
            df['Publisher'] = df['Publisher'].fillna('unknown')
        
        return df
    
    def process_books(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete book preprocessing pipeline
        """
        # Drop books with missing descriptions
        df = df.dropna(subset=['Description']).copy()
        
        # Clean descriptions
        df = self.clean_descriptions(df)
        
        # Remove books with very short descriptions
        df['length'] = df['Description'].apply(lambda x: len(str(x).split()))
        df['Description'] = df['Description'].replace(r'^\s*$', np.nan, regex=True)
        df = df.dropna(subset=['Description'])
        df = df[df['length'] >= self.min_description_word_count].copy()
        df = df.drop(columns=['length'])
        
        # Convert unknown publisher back to NaN for deduplication
        if 'Publisher' in df.columns:
            df['Publisher'] = df['Publisher'].replace('unknown', np.nan)
        
        # Drop duplicates (keep first occurrence, prioritizing non-null Publisher)
        df = df.sort_values(by='Publisher', na_position='last')\
               .drop_duplicates(subset=['Name', 'Authors', 'Description'], keep='first')
        
        # Extract series information
        df['BookSeriesInfo'] = df['Name'].apply(self.get_book_series_info)
        
        # Clean book name (remove series info)
        df['Title'] = df['Name'].apply(self.remove_series_info)
        
        # Impute missing language
        if 'Language' in df.columns:
            df.loc[df['Language'].isna(), 'Language'] = \
                df.loc[df['Language'].isna(), 'Name'].apply(self.detect_language)
        
        # Clean publisher name
        if 'Publisher' in df.columns:
            df['Publisher'] = df['Publisher'].astype(str).str.replace('"', '')
            df['Publisher'] = df['Publisher'].str.replace(' ', '_')
        
        # Transform names to single tokens
        if 'Authors' in df.columns:
            df['Authors'] = df['Authors'].str.replace(' ', '_')
        
        # Create bag of words (BOW) from metadata
        bow_cols = ['BookSeriesInfo', 'Authors', 'Publisher', 'Language']
        for col in bow_cols:
            if col not in df.columns:
                df[col] = ''
        df['bow'] = df[bow_cols].fillna('').apply(lambda x: ' '.join(x), axis=1)
        
        return df

