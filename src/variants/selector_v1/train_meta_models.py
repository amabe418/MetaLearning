from __future__ import annotations
import os
import joblib
import pandas as pd

from variants.selector_v1.meta_database import MetaDatabase
from variants.selector_v1.config import SelectorV1Config

from meta_learner import AlgorithmSelector, PerformancePredictor


def train_and_save(cfg: SelectorV1Config) -> None:
    db = MetaDatabase(cfg.store_dir)
    if not db.exists():
        raise RuntimeError(
            f"No existe meta-base en {cfg.store_dir}. "
            f"Primero corre el builder (benchmark + meta-features)."
        )

    meta_features_df, performances_df, _configs_df = db.load()

    # Entrenar (1) selector de algoritmo
    selector = AlgorithmSelector(algorithms=list(cfg.algorithms))
    selector.train(meta_features_df, performances_df)

    # Entrenar (2) predictor de rendimiento
    predictor = PerformancePredictor()
    predictor.train(meta_features_df, performances_df)

    os.makedirs(cfg.store_dir, exist_ok=True)
    joblib.dump(selector, os.path.join(cfg.store_dir, "algorithm_selector.joblib"))
    joblib.dump(predictor, os.path.join(cfg.store_dir, "performance_predictor.joblib"))


if __name__ == "__main__":
    cfg = SelectorV1Config()
    train_and_save(cfg)
    print("OK: meta-modelos entrenados y guardados.")
