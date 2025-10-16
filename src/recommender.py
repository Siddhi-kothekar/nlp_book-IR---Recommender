from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def aggregate_product_scores(
	df: pd.DataFrame,
	retrieved_indices: np.ndarray,
	retrieved_scores: np.ndarray,
	review_sentiments: np.ndarray,
	k_recommendations: int = 10,
	sentiment_weight: float = 0.5,
	rating_weight: float = 0.5,
	sentiment_min: float = -1.0,
	rating_min: float = 0.0,
) -> List[Tuple[str, float]]:
	asins = df["asin"].to_numpy()
	ratings = df["overall"].to_numpy(dtype=float)

	product_to_score: Dict[str, float] = {}
	product_to_hits: Dict[str, int] = {}

	for row_scores, row_idx in zip(retrieved_scores, retrieved_indices):
		for s, i in zip(row_scores, row_idx):
			# Clamp cosine similarity to non-negative to avoid penalizing with unrelated negatives
			sim = float(s)
			if sim <= 0.0:
				continue
			asin = asins[i]
			rating = ratings[i]
			sent = float(review_sentiments[i])
			if sent < sentiment_min or rating < rating_min:
				continue
			# Combine positive similarity with sentiment and rating
			review_score = sim * (1.0 + sentiment_weight * sent) * (1.0 + rating_weight * (rating - 3.0) / 2.0)
			product_to_score[asin] = product_to_score.get(asin, 0.0) + review_score
			product_to_hits[asin] = product_to_hits.get(asin, 0) + 1

	items: List[Tuple[str, float]] = []
	for asin, tot in product_to_score.items():
		hits = product_to_hits.get(asin, 1)
		items.append((asin, tot / float(hits)))

	# Fallback: if no items passed filters, suggest top-rated products overall
	if not items:
		rated = (
			df[["asin", "overall"]]
			.groupby("asin")
			.mean()["overall"]
			.sort_values(ascending=False)
		)
		items = [(a, float(score)) for a, score in rated.head(k_recommendations).items()]
		return items

	items.sort(key=lambda x: x[1], reverse=True)
	return items[:k_recommendations]
