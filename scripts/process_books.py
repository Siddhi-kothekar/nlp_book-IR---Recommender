"""
Book Data Processing Pipeline
Processes raw book data through cleaning, keyword extraction, and vectorization
"""
import argparse
import os
import sys
import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.book_processor import BookProcessor
from src.book_keywords import BookKeywordExtractor
from src.book_vectorizer import BookVectorizer


def main():
    parser = argparse.ArgumentParser(
        description="Process books data through preprocessing, keyword extraction, and vectorization"
    )
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", default="data/keywords.csv", help="Output path for processed keywords")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--skip-preprocessing", action="store_true", help="Skip preprocessing step")
    parser.add_argument("--skip-keywords", action="store_true", help="Skip keyword extraction step")
    parser.add_argument("--build-vectors", action="store_true", help="Build and save TF-IDF vectors")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    # Step 1: Preprocessing
    if not args.skip_preprocessing:
        print("=" * 60)
        print("STEP 1: Preprocessing Books Data")
        print("=" * 60)
        
        # Load raw data
        print(f"Loading data from {args.input}...")
        books_data = pd.read_csv(
            args.input,
            usecols=['Id', 'Name', 'Authors', 'ISBN', 'PublishYear', 
                     'Publisher', 'Language', 'Description']
        )
        print(f"Loaded {len(books_data)} books")
        
        # Process books
        processor = BookProcessor(
            min_description_word_count=cfg.get("min_description_word_count", 3)
        )
        books_data = processor.process_books(books_data)
        
        # Save preprocessed data
        preprocessed_path = "data/preprocessed.csv"
        os.makedirs("data", exist_ok=True)
        books_data.to_csv(preprocessed_path, index=False)
        print(f"Saved preprocessed data to {preprocessed_path}")
        print(f"Final dataset size: {len(books_data)} books")
    else:
        print("Skipping preprocessing step...")
        preprocessed_path = args.input
    
    # Step 2: Keyword Extraction
    if not args.skip_keywords:
        print("\n" + "=" * 60)
        print("STEP 2: Keyword Extraction using KeyBERT")
        print("=" * 60)
        
        # Load preprocessed data
        fe_data = pd.read_csv(preprocessed_path, usecols=["Id", "Name", "Language", "Description", "bow"])
        
        # Extract keywords
        extractor = BookKeywordExtractor(
            model_name=cfg.get("keybert_model", "all-MiniLM-L6-v2")
        )
        
        fe_data = extractor.process_books(
            fe_data,
            description_col="Description",
            bow_col="bow",
            top_n=cfg.get("keywords_top_n", 10)
        )
        
        # Save keywords
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        fe_data.to_csv(args.output, index=False)
        print(f"Saved keywords to {args.output}")
        print(f"Books with keywords: {len(fe_data)}")
    else:
        print("Skipping keyword extraction step...")
    
    # Step 3: Vectorization (optional)
    if args.build_vectors:
        print("\n" + "=" * 60)
        print("STEP 3: Building TF-IDF Vectors and Similarity Matrix")
        print("=" * 60)
        
        # Load keywords
        model_data = pd.read_csv(args.output)
        
        # Build vectorizer
        vectorizer = BookVectorizer(
            min_df=cfg.get("min_df", 3),
            max_df=cfg.get("max_df", 0.6)
        )
        
        # Fit and transform
        vectorizer.fit_transform(model_data["keywords"])
        
        # Compute similarity
        similarity_matrix = vectorizer.compute_similarity()
        
        # Save vectors
        artifacts_dir = cfg.get("artifacts_dir", "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)
        vectors_path = os.path.join(artifacts_dir, "book_vectors.joblib")
        vectorizer.save_vectors(vectors_path)
        
        # Save metadata
        metadata_path = os.path.join(artifacts_dir, "book_metadata.json")
        import json
        metadata = {
            "names": model_data["Name"].tolist(),
            "ids": model_data["Id"].tolist(),
            "authors": model_data.get("Authors", [""] * len(model_data)).tolist(),
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)
        
        print(f"Saved vectors to {vectors_path}")
        print(f"Saved metadata to {metadata_path}")
        print(f"Similarity matrix shape: {similarity_matrix.shape}")
    
    print("\n" + "=" * 60)
    print("✅ Processing Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

