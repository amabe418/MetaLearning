"""
Utilities to extract meta-features from OpenML datasets using amltk.
"""

from __future__ import annotations

from typing import Dict

import openml
import pandas as pd
from amltk.metalearning import DatasetStatistic, MetaFeature, compute_metafeatures


class NAValues(DatasetStatistic):
    """Mask of NA values in a dataset."""

    @classmethod
    def compute(
        cls,
        x: pd.DataFrame,
        y: pd.Series | pd.DataFrame,
        dependancy_values: dict,
    ) -> pd.DataFrame:
        return x.isna()


class PercentageNA(MetaFeature):
    """Percentage of missing values."""

    dependencies = (NAValues,)

    @classmethod
    def compute(
        cls,
        x: pd.DataFrame,
        y: pd.Series | pd.DataFrame,
        dependancy_values: dict,
    ) -> float:
        na_values = dependancy_values[NAValues]
        n_na = na_values.sum().sum()
        n_values = int(x.shape[0] * x.shape[1])
        return float(n_na / n_values) if n_values else 0.0


def _to_dense(df: pd.DataFrame) -> pd.DataFrame:
    """Convert sparse columns (from OpenML) to dense to avoid skew errors."""
    return df.apply(lambda col: col.sparse.to_dense() if hasattr(col, "sparse") else col)


def extract_meta_features(X: pd.DataFrame, y: pd.Series | pd.DataFrame) -> Dict:
    """
    Extrae meta-caracteristicas usando amltk. Convierte columnas dispersas a densas.
    """
    X_dense = _to_dense(X)
    return compute_metafeatures(X_dense, y)


def extract_meta_features_batch(datasets: pd.DataFrame) -> pd.DataFrame:
    """
    Extrae meta-caracteristicas para un lote de datasets (columnas deben incluir 'dataset_id').
    """
    meta_features_list = []

    for _, row in datasets.iterrows():
        dataset_id = row["dataset_id"]
        dataset = openml.datasets.get_dataset(dataset_id, download_data=True)
        X, y, _, _ = dataset.get_data(
            dataset_format="dataframe",
            target=dataset.default_target_attribute,
        )
        mfs = extract_meta_features(X, y)
        mfs["dataset_id"] = dataset_id
        meta_features_list.append(mfs)

    return pd.DataFrame(meta_features_list)
