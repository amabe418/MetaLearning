from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd


@dataclass
class MetaDatabase:
    """
    Almacena el meta-dataset:
      - meta_features_df: filas = dataset_id, cols = meta-features
      - performances_df: filas = dataset_id, cols = algoritmos (accuracy u otra métrica)
      - configs_df: (opcional) mejores configs por algoritmo/dataset si luego quieres warm-start real
    """
    store_dir: str

    def __post_init__(self):
        os.makedirs(self.store_dir, exist_ok=True)

    @property
    def meta_features_path(self) -> str:
        return os.path.join(self.store_dir, "meta_features.parquet")

    @property
    def performances_path(self) -> str:
        return os.path.join(self.store_dir, "performances.parquet")

    @property
    def configs_path(self) -> str:
        return os.path.join(self.store_dir, "configs.parquet")

    def save(
        self,
        meta_features_df: pd.DataFrame,
        performances_df: pd.DataFrame,
        configs_df: Optional[pd.DataFrame] = None
    ) -> None:
        meta_features_df.to_parquet(self.meta_features_path, index=True)
        performances_df.to_parquet(self.performances_path, index=True)
        if configs_df is not None:
            configs_df.to_parquet(self.configs_path, index=True)

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        mf = pd.read_parquet(self.meta_features_path)
        perf = pd.read_parquet(self.performances_path)
        cfg = None
        if os.path.exists(self.configs_path):
            cfg = pd.read_parquet(self.configs_path)
        return mf, perf, cfg

    def exists(self) -> bool:
        return os.path.exists(self.meta_features_path) and os.path.exists(self.performances_path)
