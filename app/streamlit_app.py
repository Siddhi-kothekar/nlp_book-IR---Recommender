import os
# Disable TensorFlow in Transformers to avoid tf_keras import issues
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

import json
import os as _os
import numpy as np
import pandas as pd
import streamlit as st
import yaml

from src.embeddings import EmbeddingModel
from src.indexer import FaissIndex
from src.recommender import aggregate_product_scores
from src.data import load_reviews_csv, clean_reviews, remap_columns
from src.sentiment import VaderSentiment

try:
	from src.reranker import ReviewReranker  # optional import
	_HAS_RERANKER = True
except Exception:
	_HAS_RERANKER = False


@st.cache_resource
def load_resources(cfg_path: str):
	with open(cfg_path, "r", encoding="utf-8") as f:
		cfg = yaml.safe_load(f)
	art_dir = cfg.get("artifacts_dir", "artifacts")
	index_path = _os.path.join(art_dir, "reviews.index.faiss")
	meta_path = _os.path.join(art_dir, "reviews.meta.json")
	os.makedirs(art_dir, exist_ok=True)

	meta = None
	try:
		with open(meta_path, "r", encoding="utf-8") as f:
			meta = json.load(f)
	except Exception:
		data_path = cfg.get("data_path", "data/sample_amazon_reviews.csv")
		st.warning(f"Metadata not found or invalid. Building from CSV: {data_path}")
		df_raw = load_reviews_csv(data_path)
		df = remap_columns(df_raw, cfg.get("column_map", {}))
		df = clean_reviews(df, max_review_length=int(cfg.get("max_review_length", 512)))
		sent_model = VaderSentiment()
		df["sentiment"] = sent_model.score_texts(df["reviewText"].tolist())
		meta = {
			"columns": df.columns.tolist(),
			"asin": df["asin"].tolist(),
			"title": df.get("title", pd.Series([""] * len(df))).tolist(),
			"overall": df["overall"].astype(float).tolist(),
			"sentiment": df["sentiment"].astype(float).tolist(),
			"reviewText": df["reviewText"].astype(str).tolist(),
		}
		with open(meta_path, "w", encoding="utf-8") as f:
			json.dump(meta, f)

	try:
		index = FaissIndex.load(index_path)
	except Exception:
		emb_model = EmbeddingModel(cfg.get("model_name", "sentence-transformers/all-mpnet-base-v2"))
		reviews = meta.get("reviewText") or []  # type: ignore[union-attr]
		embs = emb_model.encode(reviews, batch_size=int(cfg.get("batch_size", 256)))
		index = FaissIndex.from_embeddings(embs)
		index.save(index_path)
		return cfg, index, meta, emb_model

	emb_model = EmbeddingModel(cfg.get("model_name", "sentence-transformers/all-mpnet-base-v2"))
	return cfg, index, meta, emb_model


def main():
	st.title("Amazon Reviews IR + Recommender")
	cfg, index, meta, emb_model = load_resources("config.yaml")

	query = st.text_input("Enter product name or need (e.g., 'table for study'):")
	top_k = st.slider("Top K Reviews", 10, 200, int(cfg.get("top_k", 50)), step=10)
	n_recs = st.slider("Number of product recommendations", 3, 20, int(cfg.get("recommendations", 10)))
	reviews_to_show = st.slider("Show top N positive reviews", 3, 50, int(cfg.get("reviews_to_show", 10)))
	sentiment_weight = st.slider("Sentiment weight", 0.0, 1.0, 0.6, 0.05)
	rating_weight = st.slider("Rating weight", 0.0, 1.0, 0.4, 0.05)
	sentiment_min = st.slider("Minimum sentiment (filter)", -1.0, 1.0, float(cfg.get("sentiment_threshold", 0.2)), 0.05)
	rating_min = st.slider("Minimum rating (filter)", 1, 5, int(cfg.get("min_rating", 4)))
	rerank = st.checkbox("Use cross-encoder reranker (more accurate, slower)", value=False if not _HAS_RERANKER else True, disabled=not _HAS_RERANKER)

	if st.button("Search & Recommend") and query:
		q_emb = emb_model.encode([query])
		scores, idx = index.search(q_emb, top_k=top_k)

		asin = np.array(meta["asin"])  # type: ignore[index]
		title = np.array(meta["title"])  # type: ignore[index]
		overall = np.array(meta["overall"], dtype=float)  # type: ignore[index]
		sentiment = np.array(meta["sentiment"], dtype=float)  # type: ignore[index]
		review_texts = np.array(meta.get("reviewText", [""] * len(asin)), dtype=object)  # type: ignore[index]

		# Build a list of retrieved review candidates
		flat_indices = idx[0].tolist()
		cands = [str(review_texts[i]) for i in flat_indices]
		cand_asin = [str(asin[i]) for i in flat_indices]
		cand_score = [float(scores[0][j]) for j in range(len(flat_indices))]

		# Optional reranking
		if rerank and _HAS_RERANKER:
			try:
				reranker = ReviewReranker(cfg.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
				re_scores = reranker.score(query, cands)
				order = np.argsort(re_scores)[::-1]
				flat_indices = [flat_indices[i] for i in order]
				cands = [cands[i] for i in order]
				cand_asin = [cand_asin[i] for i in order]
				cand_score = [float(re_scores[i]) for i in order]
			except Exception as e:
				st.warning(f"Reranker failed: {e}")

		# Show top positive reviews matching the query
		st.subheader("Top Reviews for your query")
		shown = 0
		for j, i in enumerate(flat_indices):
			if shown >= reviews_to_show:
				break
			if overall[i] >= float(rating_min) and sentiment[i] >= float(sentiment_min) and cand_score[j] > 0:
				st.write(f"- [{title[i]}] {review_texts[i]}")
				shown += 1
		if shown == 0:
			st.info("No positive reviews matched filters. Try lowering thresholds.")

		# Then aggregate to product recommendations
		df = pd.DataFrame({
			"asin": asin,
			"title": title,
			"overall": overall,
		})
		recs = aggregate_product_scores(
			df=df,
			retrieved_indices=np.array([np.array(flat_indices)]),
			retrieved_scores=np.array([np.array(cand_score)]),
			review_sentiments=sentiment,
			k_recommendations=n_recs,
			sentiment_weight=sentiment_weight,
			rating_weight=rating_weight,
			sentiment_min=float(sentiment_min),
			rating_min=float(rating_min),
		)

		st.subheader("Recommended Products")
		for asin_id, score in recs:
			titles = df.loc[df["asin"] == asin_id, "title"].unique()
			title_txt = titles[0] if len(titles) > 0 else "(No title)"
			st.markdown(f"**{title_txt}**")
			st.caption(f"ASIN: {asin_id} | Score: {score:.4f}")


if __name__ == "__main__":
	main()
