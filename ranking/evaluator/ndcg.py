"""
ndcg_evaluation.py

Este módulo proporciona herramientas para evaluar rankings de algoritmos
usando la métrica **Normalized Discounted Cumulative Gain (NDCG)**.

Objetivo:
----------
Comparar rankings predichos por diferentes recommenders o modelos con un
ranking ideal (ground-truth) de algoritmos para un dataset. La métrica
NDCG permite valorar no solo qué algoritmos aparecen en el ranking, sino
también su orden relativo, premiando que los mejores algoritmos estén en
las primeras posiciones.

Componentes principales:
-----------------------
1. **RankingConverter**:
   - Convierte un ranking predicho en un vector de relevancias según
     el ranking ideal.
   - Asigna relevancias mayores a posiciones más altas en el ranking ideal.

2. **NDCG**:
   - Calcula el **Discounted Cumulative Gain (DCG)** de un ranking.
   - Normaliza con el DCG del ranking ideal (**IDCG**) para obtener NDCG ∈ [0,1].

3. **sanity_check**:
   - Permite verificar que la implementación manual de NDCG coincida con
     `sklearn.metrics.ndcg_score`.
"""


from ranking.evaluator.base import RankingEvaluator
import numpy as np
from typing import List, Dict
from sklearn.metrics import ndcg_score


class NDCGEvaluator(RankingEvaluator):
    """Calcula NDCG usando relevancias derivadas del ranking ideal."""

    @staticmethod
    def ranking_to_relevance(predicted: List[str], ideal: List[str]) -> List[int]:
        relevance_map: Dict[str,int] = {
            alg: len(ideal) - i for i, alg in enumerate(ideal)
        }

        return [relevance_map[a] for a in predicted]
    
    def score(self, predicted: List[str], ideal: List[str]) -> float:
        rel_pred = self.ranking_to_relevance(predicted, ideal)
        rel_ideal = sorted(rel_pred, reverse=True)
        dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(rel_pred))
        idcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(rel_ideal))
                
        return dcg / idcg if idcg != 0 else 0.0



def sanity_check(predicted, ideal):
    evaluator = NDCGEvaluator()
    
    manual = evaluator.score(predicted, ideal)

    y_true = np.array([[len(ideal) - i for i in range(len(ideal))]])
    score_map = {alg: len(predicted) - i for i, alg in enumerate(predicted)}
    y_score = np.array([[score_map[a] for a in ideal]])

    skl = ndcg_score(y_true, y_score)

    return manual, skl



if __name__ == "__main__":
    # Ejemplo concreto
    ideal = ["A", "B", "D", "C"]
    predicted = ["B", "D", "A", "C"]

    manual, skl = sanity_check(predicted, ideal)

    print("Predicted ranking:", predicted)
    print("Ideal ranking     :", ideal)
    print(f"Manual NDCG       : {manual:.4f}")
    print(f"Sklearn NDCG      : {skl:.4f}")