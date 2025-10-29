import os
# Disable TensorFlow BEFORE any imports
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import sys
import json
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data import load_reviews_csv, clean_reviews, remap_columns
from src.sentiment import VaderSentiment
from src.embeddings import EmbeddingModel
from src.indexer import FaissIndex


def main() -> None:
	parser = argparse.ArgumentParser(description="Build FAISS or NumPy index from Amazon reviews CSV")
	parser.add_argument("--data", required=True, help="Path to reviews CSV")
	parser.add_argument("--artifacts_dir", default=None, help="Directory to save index and metadata")
	parser.add_argument("--config", default="config.yaml", help="Path to config file")
	args = parser.parse_args()

	with open(args.config, "r", encoding="utf-8") as f:
		cfg = yaml.safe_load(f)

	artifacts_dir = args.artifacts_dir or cfg.get("artifacts_dir", "artifacts")
	os.makedirs(artifacts_dir, exist_ok=True)

	print("Loading reviews...")
	df_raw = load_reviews_csv(args.data)
	df = remap_columns(df_raw, cfg.get("column_map", {}))
	df = clean_reviews(df, max_review_length=int(cfg.get("max_review_length", 512)))

	print("Scoring sentiments...")
	sent_model = VaderSentiment()
	df["sentiment"] = sent_model.score_texts(df["reviewText"].tolist())

	print("Encoding embeddings...")
	emb_model = EmbeddingModel(cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"))
	embs = emb_model.encode(df["reviewText"].tolist(), batch_size=int(cfg.get("batch_size", 256)))

	print("Building index...")
	index = FaissIndex(embs.shape[1])
	index.add(embs.astype("float32"))

	index_path = os.path.join(artifacts_dir, "reviews.index.faiss")
	meta_path = os.path.join(artifacts_dir, "reviews.meta.json")
	emb_path = os.path.join(artifacts_dir, "reviews.embeddings.npy")

	index.save(index_path)
	np.save(emb_path, embs)

	print("Saving metadata...")
	metadata = {
		"columns": df.columns.tolist(),
		"asin": df["asin"].tolist(),
		"title": df.get("title", pd.Series([""] * len(df))).tolist(),
		"overall": df["overall"].astype(float).tolist(),
		"sentiment": df["sentiment"].astype(float).tolist(),
		"reviewText": df["reviewText"].astype(str).tolist(),
	}
	with open(meta_path, "w", encoding="utf-8") as f:
		json.dump(metadata, f)

	print(f"Saved index to {index_path}")
	print(f"Saved embeddings to {emb_path}")
	print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
	main()
