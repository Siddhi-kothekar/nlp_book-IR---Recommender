from typing import Tuple
import os
import numpy as np

try:
	import faiss  # type: ignore
	_HAS_FAISS = True
except Exception:
	_HAS_FAISS = False


class FaissIndex:
	def __init__(self, dim: int) -> None:
		self.dim = dim
		self._backend = "faiss" if _HAS_FAISS else "numpy"
		self._embeddings: np.ndarray | None = None
		if self._backend == "faiss":
			self.index = faiss.IndexFlatIP(dim)  # type: ignore[attr-defined]
		else:
			self.index = None

	@classmethod
	def from_embeddings(cls, embeddings: np.ndarray) -> "FaissIndex":
		obj = cls(embeddings.shape[1])
		obj._backend = "numpy"
		obj._embeddings = embeddings.astype('float32')
		obj.index = None
		return obj

	def add(self, embeddings: np.ndarray) -> None:
		if self._backend == "faiss":
			self.index.add(embeddings.astype('float32'))  # type: ignore[union-attr]
		else:
			# store normalized embeddings for cosine via dot
			self._embeddings = embeddings.astype('float32')

	def search(self, queries: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
		if self._backend == "faiss":
			scores, idx = self.index.search(queries.astype('float32'), top_k)  # type: ignore[union-attr]
			return scores, idx
		else:
			if self._embeddings is None:
				raise RuntimeError("Embeddings not loaded for numpy backend")
			# queries and embeddings are L2-normalized, cosine = dot
			sim = queries.astype('float32') @ self._embeddings.T
			# top-k per row
			idx = np.argpartition(-sim, kth=min(top_k, sim.shape[1]-1), axis=1)[:, :top_k]
			# sort top-k
			row_indices = np.arange(sim.shape[0])[:, None]
			sorted_order = np.argsort(-sim[row_indices, idx])
			idx = idx[row_indices, sorted_order]
			scores = sim[row_indices, idx]
			return scores, idx

	def save(self, path: str) -> None:
		os.makedirs(os.path.dirname(path), exist_ok=True)
		if self._backend == "faiss":
			faiss.write_index(self.index, path)  # type: ignore[union-attr]
		else:
			# write backend marker and persist embeddings alongside
			with open(path + ".backend", "w", encoding="utf-8") as f:
				f.write("numpy")
			art_dir = os.path.dirname(path)
			emb_path = os.path.join(art_dir, "reviews.embeddings.npy")
			if self._embeddings is not None:
				np.save(emb_path, self._embeddings)

	@staticmethod
	def load(path: str) -> "FaissIndex":
		backend = "faiss" if _HAS_FAISS else "numpy"
		# if a .backend marker exists, respect it
		marker = path + ".backend"
		if os.path.exists(marker):
			with open(marker, "r", encoding="utf-8") as f:
				backend = f.read().strip() or backend
		if backend == "faiss":
			index = faiss.read_index(path)  # type: ignore[attr-defined]
			obj = FaissIndex(index.d)  # type: ignore[attr-defined]
			obj.index = index
			obj._backend = "faiss"
			return obj
		# numpy backend: load embeddings from sibling npy
		art_dir = os.path.dirname(path)
		emb_path = os.path.join(art_dir, "reviews.embeddings.npy")
		if not os.path.exists(emb_path):
			raise FileNotFoundError(f"Embeddings not found for numpy backend at {emb_path}")
		embs = np.load(emb_path).astype('float32')
		obj = FaissIndex.from_embeddings(embs)
		return obj
