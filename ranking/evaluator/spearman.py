"""
spearman.py

Este módulo proporciona herramientas para evaluar rankings de algoritmos
usando la métrica **Spearmans rank correlation**.

Objetivo:
----------
Comparar rankings predichos por diferentes recommenders o modelos con un
ranking ideal (ground-truth) de algoritmos para un dataset. Spearman
evalúa la correlación entre los órdenes de los elementos, premiando que
los mejores algoritmos aparezcan en posiciones similares.

Componentes principales:
-----------------------
1. **RankingConverter**:
   - Convierte rankings en vectores de posiciones numéricas.
   - Idealmente, la primera posición recibe valor 1, la segunda 2, etc.

2. **SpearmanEvaluator**:
   - Calcula la correlación de Spearman entre ranking predicho y ideal.

3. **sanity_check**:
   - Permite verificar que la implementación manual coincida con
     `scipy.stats.spearmanr`.
"""


from ranking.evaluator.base import RankingEvaluator
from typing import List
from scipy.stats import spearmanr

class SpearmanEvaluator(RankingEvaluator):
    """Calcula Spearman r usando posiciones numéricas."""

    @staticmethod
    def ranking_to_positions(ranking: List[str]) -> List[int]:
        return [ranking.index(item) + 1 for item in ranking]

    def score(self, predicted: List[str], ideal: List[str]) -> float:
        pos_pred = [predicted.index(a) + 1 for a in ideal]  # align predicted to ideal
        pos_ideal = list(range(1, len(ideal) + 1))
        
        corr, _ = spearmanr(pos_pred, pos_ideal)
        return corr
    


def sanity_check(predicted: List[str], ideal: List[str]):
    """
    Compare manual implementation with scipy spearmanr.
    """
    evaluator = SpearmanEvaluator()
    manual = evaluator.score(predicted, ideal)
    pos_pred = [predicted.index(a) + 1 for a in ideal]  # align predicted to ideal
    pos_ideal = list(range(1, len(ideal) + 1))

    scipy_val, _ = spearmanr(pos_pred, pos_ideal)

    return manual, scipy_val



if __name__ == "__main__":
    # Ejemplo concreto
    ideal = ["A", "B", "C", "D"]
    predicted = ["C", "D", "B", "A"]

    manual, scipy_val = sanity_check(predicted, ideal)

    print("Predicted ranking:", predicted)
    print("Ideal ranking     :", ideal)
    print(f"Manual Spearman   : {manual:.4f}")
    print(f"Scipy Spearman    : {scipy_val:.4f}")


