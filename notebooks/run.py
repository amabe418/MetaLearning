import openml
import pandas as pd
from openml.tasks import TaskType

def get_algorithm_ranking_for_dataset(
    dataset_id,
    metric="accuracy",
):
    """
    Returns a ranking of algorithms for a given dataset
    based on OpenML runs.
    """

    # 1. Obtener tareas del dataset
    tasks = openml.tasks.list_tasks(
        data_id=dataset_id,
        task_type= TaskType.SUPERVISED_CLASSIFICATION,
        output_format="dataframe"
    )

    if tasks.empty:
        return None

    task_ids = tasks.index.tolist()

    # 2. Obtener runs asociados a esas tareas
    runs = openml.runs.list_runs(
        task=task_ids,
        output_format="dataframe"
    )

    if runs.empty:
        return None

    # 3. Obtener información detallada de cada run para extraer las métricas
    detailed_runs = []
    for run_id in runs["run_id"].head(100):  # Limitar a primeros 100 para no sobrecargar
        try:
            run = openml.runs.get_run(run_id)
            # Extraer el flow name
            flow_name = run.flow_name if hasattr(run, 'flow_name') else "Unknown"
            # Las evaluaciones están en run.evaluations
            score = None
            if hasattr(run, 'evaluations') and metric in run.evaluations:
                score = run.evaluations[metric].value
            
            if score is not None:
                detailed_runs.append({
                    "run_id": run_id,
                    "flow_name": flow_name,
                    "score": score
                })
        except Exception as e:
            continue

    if not detailed_runs:
        return None

    # 4. Crear DataFrame
    df = pd.DataFrame(detailed_runs)

    # 5. Agregar: mejor score por algoritmo
    agg = df.groupby("flow_name")["score"].max().reset_index()
    agg.columns = ["algorithm", "score"]

    # 6. Ranking (mayor score = mejor)
    ranking = agg.sort_values("score", ascending=False).reset_index(drop=True)
    ranking["rank"] = ranking.index + 1

    return ranking



dataset_id = 61  # Iris dataset
ranking = get_algorithm_ranking_for_dataset(dataset_id)

if ranking is not None:
    print(ranking.head())
else:
    print("No ranking found for this dataset")
