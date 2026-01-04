import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def l1_distance(a: np.ndarray, b: np.ndarray, a_nan: np.ndarray, b_nan: np.ndarray) -> float:
    # If both NaN -> distance 0 for that feature; if only one NaN -> distance 1.
    both_nan = a_nan & b_nan
    one_nan = a_nan ^ b_nan

    diff = np.abs(a - b)
    diff[both_nan] = 0.0
    diff[one_nan] = 1.0

    return float(np.sum(diff))

def knn_for_target(
    target_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    """
    target_df: DataFrame con UNA fila (dataset objetivo)
    pool_df: DataFrame con datasets históricos
    """

    # Extraer ID del target
    target_id = target_df.iloc[0]["openml_task"]
    
    print(target_id)

    # Excluir columnas no numéricas: identificador, flow y métrica de rendimiento
    exclude_cols = ["openml_task", "flow", "area_under_roc_curve"]
    target_features = target_df.drop(columns=exclude_cols).to_numpy(dtype=float)[0]
    pool_ids = pool_df["openml_task"].tolist()
    pool_features = pool_df.drop(columns=exclude_cols).to_numpy(dtype=float)

    # NaN masks
    target_nan = np.isnan(target_features)
    pool_nan = np.isnan(pool_features)

    dists = []

    for i in range(len(pool_ids)):
        dist = l1_distance(
            target_features,
            pool_features[i],
            target_nan,
            pool_nan[i],
        )
        dists.append((pool_ids[i], dist))

    # Ordenar por distancia
    dists.sort(key=lambda x: x[1])

    # Agrupar por openml_task: tomar solo uno de cada dataset (todos tienen mismas meta-features)
    seen_tasks = set()
    unique_dists = []
    for task_id, dist in dists:
        if task_id not in seen_tasks:
            seen_tasks.add(task_id)
            unique_dists.append((task_id, dist))
    
    # Top-K de datasets únicos
    rows = []
    for rank, (neighbor_id, dist) in enumerate(unique_dists[:k], start=1):
        rows.append({
            "target_dataset": target_id,
            "neighbor_id": neighbor_id,
            "distance_l1": dist,
            "rank": rank,
        })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="KNN for a target dataset using L1 distance on meta-features"
    )
    parser.add_argument("--target-csv", type=str, required=True,
                        help="CSV with meta-features of the target dataset (1 row)")
    parser.add_argument("--pool-csv", type=str, required=True,
                        help="CSV with historical datasets meta-features")
    parser.add_argument("--out-csv", type=str, required=True,
                        help="Output CSV with K nearest neighbors")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of neighbors")

    args = parser.parse_args()

    target_df = pd.read_csv(Path(args.target_csv))
    pool_df = pd.read_csv(Path(args.pool_csv))

    for df, name in [(target_df, "target"), (pool_df, "pool")]:
        if "openml_task" not in df.columns:
            raise RuntimeError(f"{name} CSV must include 'openml_task'")

    if len(target_df) != 1:
        raise RuntimeError("Target CSV must contain exactly one dataset")

    knn_df = knn_for_target(
        target_df=target_df,
        pool_df=pool_df,
        k=args.k
    )

    knn_df.to_csv(Path(args.out_csv), index=False)
    print(f"[OK] Neighbors saved to {args.out_csv}")


if __name__ == "__main__":
    main()
