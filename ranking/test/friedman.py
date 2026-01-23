"""
friedman_ranking.py

Módulo para evaluar modelos de recomendación de algoritmos usando métricas de ranking
(NDCG o Spearman) y aplicar el test de Friedman.

Objetivo:
----------
Comparar varios modelos/recomendadores sobre múltiples datasets, verificando
si existen diferencias significativas en la calidad de los rankings que generan.

Flujo:
1. Cada modelo produce un ranking de algoritmos para cada dataset.
2. Se calcula la métrica de comparación con el ranking ideal (NDCG o Spearman).
3. Se construye una matriz dataset × modelo con los scores.
4. Se aplica el test de Friedman para evaluar diferencias globales.
5. (Opcional) Post-hoc para identificar qué modelos difieren.
"""

import pandas as pd
from scipy.stats import friedmanchisquare
from ranking.evaluator.ndcg import NDCGEvaluator


# -----------------------------
# Friedman
# -----------------------------

class FriedmanTest:
    """Aplica el test de Friedman sobre un DataFrame de scores."""

    @staticmethod
    def apply(df_scores: pd.DataFrame):
        """
        Parameters
        ----------
        df_scores : DataFrame
            Filas = datasets, Columnas = modelos, Valores = score por dataset

        Returns
        -------
        statistic, p-value
        """
        stat, p = friedmanchisquare(*[df_scores[col].values for col in df_scores.columns])
        return stat, p

    @staticmethod
    def ranks(df_scores: pd.DataFrame) -> pd.DataFrame:
        """
        Convierte los scores a ranks por dataset (fila).

        Ejemplo: fila [0.8, 0.6, 0.9] → ranks [2, 3, 1] (1 = mejor)
        """
        return df_scores.rank(ascending=False, method="average", axis=1)

# -----------------------------
# Ejemplo de uso
# -----------------------------

if __name__ == "__main__":
    # Supongamos que tenemos 3 modelos y 4 datasets
    datasets = ["d1", "d2", "d3", "d4"]
    models = ["Random", "Baseline", "Metabu"]

    # Rankings simulados de algoritmos para cada dataset y modelo
    # Normalmente aquí usarías tus recommenders reales
    ideal_ranking = ["A", "B", "C", "D"]
    model_rankings = {
        "Random": [
            ["B", "D", "A", "C"],
            ["C", "A", "D", "B"],
            ["A", "C", "B", "D"],
            ["D", "B", "A", "C"]
        ],
        "Baseline": [
            ["A", "B", "C", "D"],
            ["A", "B", "D", "C"],
            ["B", "A", "C", "D"],
            ["A", "C", "B", "D"]
        ],
        "Metabu": [
            ["A", "C", "B", "D"],
            ["A", "B", "C", "D"],
            ["A", "B", "C", "D"],
            ["A", "B", "D", "C"]
        ]
    }

    # Elegimos la métrica
    evaluator = NDCGEvaluator()  # o SpearmanEvaluator()

    # Calculamos scores
    scores = {
        model: [evaluator.score(r, ideal_ranking) for r in rankings]
        for model, rankings in model_rankings.items()
    }

    print(scores)

    df_scores = pd.DataFrame(scores, index=datasets)
    print("Scores por dataset y modelo:")
    print(df_scores)

    # Ranks por dataset
    df_ranks = FriedmanTest.ranks(df_scores)
    print("\nRanks por dataset (1 = mejor):")
    print(df_ranks)

    # Test de Friedman
    stat, p = FriedmanTest.apply(df_scores)
    print(f"\nFriedman statistic = {stat:.4f}, p-value = {p:.4f}")
    if p < 0.05:
        print("→ Hay diferencias significativas entre los modelos")
    else:
        print("→ No hay diferencias significativas entre los modelos")
