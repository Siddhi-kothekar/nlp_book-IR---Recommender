from typing import Iterable
import numpy as np
from sentence_transformers import SentenceTransformer


def l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
	# Avoid division by zero
	norms = np.linalg.norm(matrix, axis=1, keepdims=True)
	norms = np.maximum(norms, eps)
	return matrix / norms


class EmbeddingModel:
	def __init__(self, model_name: str) -> None:
		self.model = SentenceTransformer(model_name)

	def encode(self, texts: Iterable[str], batch_size: int = 256) -> np.ndarray:
		embs = self.model.encode(
			list(texts),
			batch_size=batch_size,
			convert_to_numpy=True,
			normalize_embeddings=False,
			show_progress_bar=True,
		)
		return l2_normalize(embs)
