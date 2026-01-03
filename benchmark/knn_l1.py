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


def compute_knn(
    df: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    ids = df["dataset_id"].tolist()
    features = df.drop(columns=["dataset_id"])
    values = features.to_numpy(dtype=float)
    nan_mask = np.isnan(values)

    rows = []
    n = len(ids)
    for i in range(n):
        dists = []
        for j in range(n):
            if i == j:
                continue
            dist = l1_distance(values[i], values[j], nan_mask[i], nan_mask[j])
            dists.append((ids[j], dist))
        dists.sort(key=lambda x: x[1])
        for rank, (neighbor_id, dist) in enumerate(dists[:k], start=1):
            rows.append({
                "dataset_id": ids[i],
                "neighbor_id": neighbor_id,
                "distance_l1": dist,
                "rank": rank,
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="KNN with L1 distance for meta-features")
    parser.add_argument("--in-csv", type=str, required=True, help="Input normalized metafeatures CSV")
    parser.add_argument("--out-csv", type=str, required=True, help="Output neighbors CSV")
    parser.add_argument("--k", type=int, default=5, help="Number of neighbors")
    args = parser.parse_args()

    df = pd.read_csv(Path(args.in_csv))
    if "dataset_id" not in df.columns:
        raise RuntimeError("Input CSV must include dataset_id")

    knn_df = compute_knn(df, k=args.k)
    knn_df.to_csv(Path(args.out_csv), index=False)
    print(f"[OK] Output: {args.out_csv}")


if __name__ == "__main__":
    main()
