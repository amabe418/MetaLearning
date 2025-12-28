from __future__ import annotations
import os
import joblib
from typing import Dict, Any, List
import pandas as pd

from variants.selector_v1.config import SelectorV1Config
from variants.selector_v1.metafeature_schema import MetaFeatureSchema, compute_basic_metafeatures


def recommend_configs_for_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: SelectorV1Config,
    top_k: int = 3
) -> List[Dict[str, Any]]:
    # 1) meta-features (Sección 2 → representación del dataset)
    schema = MetaFeatureSchema.default()
    raw = compute_basic_metafeatures(X, y)
    meta = schema.enforce(raw)

    # 2) cargar predictor entrenado
    predictor_path = os.path.join(cfg.store_dir, "performance_predictor.joblib")
    if not os.path.exists(predictor_path):
        raise RuntimeError("No existe el predictor entrenado. Ejecuta train_meta_models.py")

    predictor = joblib.load(predictor_path)

    # 3) predecir performance por algoritmo
    pred = predictor.predict(meta)  # dict alg->score

    # 4) ordenar y construir CONFIGS
    ranked = sorted(pred.items(), key=lambda x: x[1], reverse=True)[:top_k]

    recs = []
    for alg, expected in ranked:
        recs.append({
            "algorithm": alg,
            "hyperparams": cfg.base_hyperparams.get(alg, {}),
            "expected_performance": float(expected),
            "source": "performance_predictor",
        })

    return recs
