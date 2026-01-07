import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _drop_low_variance(df: pd.DataFrame, eps: float) -> list[str]:
    keep = []
    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            continue
        if float(series.var()) > eps:
            keep.append(col)
    return keep


def _drop_high_corr(df: pd.DataFrame, threshold: float) -> list[str]:
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = {col for col in upper.columns if (upper[col] > threshold).any()}
    return [c for c in df.columns if c not in to_drop]


def select_metafeatures(
    df: pd.DataFrame,
    max_features: int,
    var_eps: float,
    corr_threshold: float,
) -> list[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "dataset_id"]

    # Step 1: drop low-variance / empty columns
    keep = _drop_low_variance(df[numeric_cols], eps=var_eps)
    work = df[keep].copy()

    # Step 2: impute for correlation calculation
    work = work.fillna(work.median(numeric_only=True))

    # Step 3: drop highly correlated columns
    keep = _drop_high_corr(work, threshold=corr_threshold)
    work = work[keep]

    # Step 4: keep top by variance (automatic size)
    variances = work.var().sort_values(ascending=False)
    if max_features > 0:
        keep = variances.head(max_features).index.tolist()
    else:
        keep = variances.index.tolist()

    return keep


def main() -> None:
    parser = argparse.ArgumentParser(description="Select non-redundant meta-features for KNN")
    parser.add_argument("--in-csv", type=str, required=True, help="Input metafeatures CSV")
    parser.add_argument("--out-csv", type=str, required=True, help="Output CSV with selected metafeatures")
    parser.add_argument("--max-features", type=int, default=0, help="Max features to keep (0 = auto)")
    parser.add_argument("--var-eps", type=float, default=1e-12, help="Variance threshold")
    parser.add_argument("--corr-threshold", type=float, default=0.95, help="Correlation threshold")
    args = parser.parse_args()

    in_path = Path(args.in_csv)
    df = pd.read_csv(in_path)

    if args.max_features <= 0:
        n_datasets = int(df.shape[0])
        args.max_features = max(10, min(30, n_datasets // 3))

    selected = select_metafeatures(
        df,
        max_features=args.max_features,
        var_eps=args.var_eps,
        corr_threshold=args.corr_threshold,
    )

    cols = ["dataset_id"] if "dataset_id" in df.columns else []
    cols += selected
    df[cols].to_csv(args.out_csv, index=False)

    print(f"[OK] Selected {len(selected)} metafeatures")
    print(f"[OK] Output: {args.out_csv}")


if __name__ == "__main__":
    main()
