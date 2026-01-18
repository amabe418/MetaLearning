"""
random_recommender.py

Este script implementa un **baseline aleatorio de recomendación de algoritmos**
para el problema de *algorithm ranking* en meta-learning.

Objetivo:
----------
Servir como un **baseline no informado** que genere, para cada dataset,
un ranking de algoritmos completamente aleatorio pero **reproducible**.
Se utiliza como referencia inferior para evaluar la calidad de otros
recomendadores más sofisticados.

Metodología:
-------------
1. El recomendador recibe únicamente el `dataset_id`, sin usar meta-features
   ni información de desempeño.
2. Se combina una semilla global con un hash estable del identificador del
   dataset para garantizar reproducibilidad.
3. Se genera una permutación aleatoria de los algoritmos candidatos,
   produciendo un ranking completo por dataset.

Uso experimental:
------------------
Los rankings generados se comparan con el ranking ideal mediante métricas
como **NDCG** o **Spearman**, y los resultados se agregan en una matriz
dataset * recomendador para su posterior análisis estadístico
(por ejemplo, test de **Friedman**).

Rol del baseline:
-----------------
Este método establece el nivel de desempeño esperado por azar y permite
verificar que los modelos propuestos superan claramente una recomendación
aleatoria.
"""


import random
import argparse
from typing import List
import hashlib


class RandomRecommender:
    def __init__(self, algorithms: List[str], seed: int | None = None):
        self.algorithms = algorithms
        self.seed = seed

    def _stable_hash(self, dataset_id: str) -> int:
        h = hashlib.md5(dataset_id.encode("utf-8")).hexdigest()
        return int(h, 16)

    def recommend(self, dataset_id: str) -> List[str]:
        base_seed = self.seed if self.seed is not None else 0
        rng = random.Random(base_seed + self._stable_hash(dataset_id))

        ranking = self.algorithms.copy()
        rng.shuffle(ranking)
        return ranking


def main():

    parser = argparse.ArgumentParser(description="Random algorithm recommender")
    parser.add_argument("--dataset_id", required=True, type=str)
    args = parser.parse_args()

    algorithms = ["Adaboost", "Random_Forest", "SVC"]

    random_rec = RandomRecommender(algorithms, seed=42)

    ranking = random_rec.recommend(args.dataset_id)

    print(f"Random ranking for dataset {args.dataset_id}:")
    print(ranking)


if __name__ == "__main__":
    main()
