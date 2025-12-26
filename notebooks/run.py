import openml
import pandas as pd
from openml.tasks import TaskType

def get_algorithm_ranking_for_dataset(dataset_id, metric="accuracy"):

    tasks = openml.tasks.list_tasks(
        data_id=dataset_id,
        task_type=TaskType.SUPERVISED_CLASSIFICATION,
        output_format="dataframe"
    )
    if tasks.empty:
        return None

    runs = openml.runs.list_runs(
        task=tasks.index.tolist(),
        output_format="dataframe"
    )
    if runs.empty:
        return None

    col = f"evaluation_measures.{metric}"

    # Caso 1: métrica como columna
    if col in runs.columns:
        runs = runs[runs[col].notna()]
        if runs.empty:
            return None

        df = runs[["flow_name", col]].rename(
            columns={"flow_name": "algorithm", col: "score"}
        )

    else:
        # No hay métricas usables
        print("Métricas disponibles:")
        print([c for c in runs.columns if c.startswith("evaluation_measures.")])
        return None

    agg = df.groupby("algorithm")["score"].max().reset_index()
    ranking = agg.sort_values("score", ascending=False).reset_index(drop=True)
    ranking["rank"] = ranking.index + 1
    ranking["metric"] = metric

    return ranking

# PROBLEMS = [3,6,9,11,12,15,31,37,44,50]

# for PROBLEM in PROBLEMS:

#     print("EL run numero:", PROBLEM)
#     ranking = get_algorithm_ranking_for_dataset(PROBLEM)

#     if ranking is None:
#         print("No hay runs con métricas válidas")
#     else:
#         print(ranking.head())


# dataset_id   nombre
# -----------  -------------------------
# 3            kr-vs-kp
# 6            letter
# 11           balance-scale
# 12           mfeat-factors
# 15           breast-w
# 31           credit-g
# 37           diabetes
# 44           spambase
# 50           tic-tac-toe

