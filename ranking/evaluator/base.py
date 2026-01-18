from typing import List

class RankingEvaluator:
    """Clase base para evaluar rankings de algoritmos."""

    def score(self, predicted: List[str], ideal: List[str]) -> float:
        """Devuelve un score comparando predicted vs ideal."""
        raise NotImplementedError
