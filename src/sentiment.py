from typing import Iterable, List
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Ensure VADER lexicon is available
try:
	_ = nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
	nltk.download('vader_lexicon', quiet=True)


class VaderSentiment:
	def __init__(self) -> None:
		self.analyzer = SentimentIntensityAnalyzer()

	def score_texts(self, texts: Iterable[str]) -> List[float]:
		scores: List[float] = []
		for text in texts:
			pol = self.analyzer.polarity_scores(text or "")
			scores.append(pol.get("compound", 0.0))
		return scores
