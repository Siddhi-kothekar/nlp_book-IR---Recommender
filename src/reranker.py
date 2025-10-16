from typing import List, Tuple
try:
	from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover
	CrossEncoder = None  # type: ignore


class ReviewReranker:
	def __init__(self, model_name: str) -> None:
		if CrossEncoder is None:
			raise ImportError("sentence-transformers CrossEncoder not available")
		self.model = CrossEncoder(model_name)

	def score(self, query: str, candidates: List[str], batch_size: int = 64) -> List[float]:
		pairs: List[Tuple[str, str]] = [(query, c) for c in candidates]
		scores = self.model.predict(pairs, batch_size=batch_size)
		return [float(s) for s in scores]
