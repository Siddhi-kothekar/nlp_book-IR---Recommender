import pandas as pd
from typing import Dict, List, Optional

REQUIRED_COLUMNS = ["asin", "title", "reviewText", "overall"]


def load_reviews_csv(path: str, usecols: Optional[List[str]] = None) -> pd.DataFrame:
	cols = usecols or REQUIRED_COLUMNS
	df = pd.read_csv(path, usecols=lambda c: True)  # read all, we'll map later
	return df


def remap_columns(df: pd.DataFrame, column_map: Dict[str, str]) -> pd.DataFrame:
	# column_map keys: id, title, rating, review, user, summary, category, brand
	mapped = pd.DataFrame()
	# product id: prefer explicit id else title as pseudo-id
	id_col = column_map.get("id")
	if id_col and id_col in df.columns:
		mapped["asin"] = df[id_col].astype(str)
	else:
		mapped["asin"] = df[column_map.get("title", "itemName")].astype(str)
	# title and review
	title_col = column_map.get("title", "itemName")
	review_col = column_map.get("review", "reviewText")
	mapped["title"] = df.get(title_col, "").astype(str)
	mapped["reviewText"] = df.get(review_col, "").astype(str)
	# rating
	rating_col = column_map.get("rating", "overall")
	mapped["overall"] = pd.to_numeric(df.get(rating_col, 0), errors="coerce").fillna(0)
	return mapped


def clean_reviews(df: pd.DataFrame, max_review_length: int = 512) -> pd.DataFrame:
	# Drop rows with empty reviews or invalid ratings
	df = df.dropna(subset=["reviewText"])  # type: ignore[assignment]
	df["overall"] = pd.to_numeric(df["overall"], errors="coerce").fillna(0)
	# Clip long reviews
	df["reviewText"] = df["reviewText"].astype(str).str.slice(0, max_review_length)
	# Fill missing titles
	if "title" in df.columns:
		df["title"] = df["title"].fillna("")
	return df.reset_index(drop=True)
