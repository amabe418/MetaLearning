import pandas as pd
import numpy as np


def mean_top_k_accuracy_per_task(
    csv_path: str,
    task_ids,
    k: int = 20
):
    """
    Parameters
    ----------
    csv_path : str
        Path to the CSV file
    task_ids : iterable
        Collection of task_ids
    k : int
        Number of top accuracies to average (default=20)

    Returns
    -------
    np.ndarray
        Array with the mean of top-k accuracies per task_id,
        in the same order as task_ids
    """
    df = pd.read_csv(csv_path)

    results = []

    for task_id in task_ids:
        task_rows = df[df["task_id"] == task_id]

        if task_rows.empty:
            results.append(np.nan)
            continue

        top_k = task_rows["predictive_accuracy"].nlargest(k)
        results.append(top_k.mean())

    return np.array(results)



def get_neighbors(csv_path: str, task_id):
    df = pd.read_csv(csv_path)

    row = df[df["task_id"] == task_id]

    if row.empty:
        raise ValueError(f"task_id {task_id} not found")

    neighbors = (
        row
        .filter(like="neighbor_")
        .values
        .flatten()
        .astype(int)
        .tolist()
    )

    return neighbors


def calculate_individual_score(mean_ids):

    mean_ids = np.asarray(mean_ids, dtype=float)
    mask = ~np.isnan(mean_ids)
    mean_ids = mean_ids[mask]

    if len(mean_ids) == 0:
        return np.nan
    
    L = len(mean_ids)
    weights = np.exp(-np.arange(L))
    return np.sum(weights * mean_ids) / np.sum(weights)


def calculate_general_score(all_neighbors,task_id, k=20):
    pipelines = ["adaboost", "random_forest", "libsvm_svc"]

    scores = []

    for pipeline in pipelines:
        target_csv = f"data/top_raw_target_representation/{pipeline}_target_representation.csv"
        # neighbors_csv = f"outputs/neighbors_task_{task_id}_{pipeline}.csv"

        # neighbors = get_neighbors(
        #     csv_path=neighbors_csv,
        #     task_id=task_id
        # )

        neighbors = all_neighbors[pipeline][task_id]


        means = mean_top_k_accuracy_per_task(
            csv_path=target_csv,
            task_ids=neighbors,
            k=k
        )

        score = calculate_individual_score(mean_ids= means)

        scores.append({
                "pipeline":pipeline,
                "score": score
        })

    return sorted(scores, key=lambda x: x["score"], reverse=True)

