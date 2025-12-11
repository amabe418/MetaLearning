"""
Utilities to pick diverse OpenML datasets and extract their meta-features.

The selection is driven by simple heuristics:
- Keep datasets under a configurable size budget (default: 1.5 GB, estimated).
- Sample a balanced mix of task types (classification / regression).
- Avoid datasets with too many rows or columns to keep downloads manageable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import openml
import pandas as pd

from data_loader import TASK_TYPES
from meta_features import extract_meta_features

DEFAULT_SIZE_LIMIT_MB = 1500
BYTES_PER_VALUE_NUMERIC = 8  # float64
BYTES_PER_VALUE_CATEGORICAL = 4  # small int/string encoded


def _approx_size_mb(
    n_rows: int,
    n_features: int,
    n_categorical_features: Optional[int] = None,
) -> float:
    """Rough size estimate assuming dense storage."""
    cat = max(int(n_categorical_features or 0), 0)
    num = max(int(n_features) - cat, 0)
    size_bytes = (num * BYTES_PER_VALUE_NUMERIC + cat * BYTES_PER_VALUE_CATEGORICAL) * max(
        int(n_rows), 0
    )
    return size_bytes / (1024**2)


def _prepare_tasks_df(tasks: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns and attach a size estimate."""
    df = tasks.rename(columns={"did": "dataset_id"}).copy()
    for col in (
        "NumberOfInstances",
        "NumberOfFeatures",
        "NumberOfNumericFeatures",
        "NumberOfSymbolicFeatures",
    ):
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0).astype(int)

    df["approx_size_mb"] = df.apply(
        lambda row: _approx_size_mb(
            row["NumberOfInstances"],
            row["NumberOfFeatures"],
            row.get("NumberOfSymbolicFeatures", 0),
        ),
        axis=1,
    )
    return df


def select_openml_datasets(
    task_types: Sequence[str] = ("Supervised Classification", "Supervised Regression"),
    per_type: int = 3,
    size_limit_mb: float = DEFAULT_SIZE_LIMIT_MB,
    min_instances: int = 100,
    max_instances: int = 200_000,
    max_features: int = 2_000,
    random_state: int = 42,
) -> List[int]:
    """
    Select a small, diverse set of datasets that fit within resource bounds.

    Returns:
        List of dataset IDs to download.
    """
    selected: List[int] = []

    for task_name in task_types:
        if task_name not in TASK_TYPES:
            raise ValueError(f"Tarea no soportada: {task_name}")

        tasks = openml.tasks.list_tasks(task_type=TASK_TYPES[task_name], output_format="dataframe")
        tasks = _prepare_tasks_df(tasks)
        tasks = tasks[
            (tasks["NumberOfInstances"] >= min_instances)
            & (tasks["NumberOfInstances"] <= max_instances)
            & (tasks["NumberOfFeatures"] <= max_features)
            & (tasks["approx_size_mb"] <= size_limit_mb)
        ]

        # Favor variety by shuffling before taking the head.
        tasks = tasks.sample(frac=1.0, random_state=random_state)
        selected.extend(tasks["dataset_id"].head(per_type).tolist())

    # Keep unique IDs while preserving order.
    unique_ids = []
    seen = set()
    for did in selected:
        if did not in seen:
            unique_ids.append(did)
            seen.add(did)

    return unique_ids


def download_and_extract_meta_features(
    dataset_ids: Iterable[int],
    output_dir: str | Path = "data",
    save_raw: bool = True,
    include_qualities: bool = True,
) -> pd.DataFrame:
    """
    Download datasets and compute meta-features for each.

    Args:
        dataset_ids: Iterable of OpenML dataset IDs.
        output_dir: Base directory to save results (raw data + meta-features).
        save_raw: If True, persist raw CSVs under data/raw/.
        include_qualities: If True, añade las qualities de OpenML al CSV final
            y guarda un archivo aparte `openml_qualities.csv`.

    Returns:
        DataFrame with meta-features (one row per dataset).
    """
    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw"
    meta_dir = output_dir / "meta_features"
    meta_dir.mkdir(parents=True, exist_ok=True)
    if save_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict] = []
    qualities_records: List[Dict] = []

    for did in dataset_ids:
        dataset = openml.datasets.get_dataset(did, download_data=True)
        X, y, _, _ = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute
        )

        if save_raw:
            safe_name = dataset.name.replace(" ", "_")
            raw_path = raw_dir / f"{did}_{safe_name}.csv"
            X.assign(target=y).to_csv(raw_path, index=False)

        meta = extract_meta_features(X, y)
        meta["dataset_id"] = did
        meta["name"] = dataset.name
        meta["n_rows"] = X.shape[0]
        meta["n_columns"] = X.shape[1]

        if include_qualities and dataset.qualities:
            # Prefija las qualities para evitar colisiones de nombres.
            for k, v in dataset.qualities.items():
                meta[f"quality_{k}"] = v
            qualities_records.append({"dataset_id": did, **dataset.qualities})

        records.append(meta)

    meta_df = pd.DataFrame(records)
    meta_path = meta_dir / "meta_features.csv"
    meta_df.to_csv(meta_path, index=False)

    if include_qualities and qualities_records:
        qualities_df = pd.DataFrame(qualities_records)
        qualities_path = meta_dir / "openml_qualities.csv"
        qualities_df.to_csv(qualities_path, index=False)

    return meta_df


if __name__ == "__main__":
    # Example: pick up to 6 datasets (3 classification + 3 regression) and extract their meta-features.
    dataset_ids = select_openml_datasets(per_type=10, size_limit_mb=DEFAULT_SIZE_LIMIT_MB)
    print(f"Datasets seleccionados: {dataset_ids}")
    df = download_and_extract_meta_features(dataset_ids)
    print(f"Meta-features guardados en data/meta_features/meta_features.csv ({len(df)} filas)")
