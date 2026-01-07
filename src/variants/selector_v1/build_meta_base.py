from __future__ import annotations

import os
import argparse
from typing import List, Dict, Any, Optional

import pandas as pd
import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Tu loader existente
from data_loader import load_openml_dataset, search_openml_datasets

# Variante nueva (archivos nuevos)
from config import SelectorV1Config
from metafeature_schema import (
    MetaFeatureSchema,
    compute_basic_metafeatures,
    as_dataframe,
)
from meta_database import MetaDatabase
from benchmark_ground_truth import GroundTruthBenchmarker


def parse_dataset_ids(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def load_datasets_openml(dataset_ids: List[int], cache_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    datasets: List[Dict[str, Any]] = []
    for did in dataset_ids:
        d = load_openml_dataset(did, cache_dir=cache_dir)
        if d is None:
            print(f"[WARN] No se pudo cargar dataset_id={did}. Se omite.")
            continue

        # Asegurar formato DataFrame/Series
        X = d["X"]
        y = d["y"]
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if isinstance(y, pd.DataFrame):
            # si viene como df, toma primera col
            y = y.iloc[:, 0]
        else:
            y = pd.Series(y)

        datasets.append({"id": d["id"], "name": d.get("name", str(d["id"])), "X": X, "y": y})

    return datasets


def compute_meta_features_table(
    datasets: List[Dict[str, Any]],
    schema: MetaFeatureSchema
) -> pd.DataFrame:
    rows = []
    index = []
    for d in datasets:
        X, y = d["X"], d["y"]
        raw = compute_basic_metafeatures(X, y)
        rows.append(raw)
        index.append(d["id"])
    mf = as_dataframe(rows, schema=schema, index=index)
    mf.index.name = "dataset_id"
    return mf.sort_index()


def main():
    parser = argparse.ArgumentParser(description="Build meta-base (meta-features + performances) for selector_v1")
    parser.add_argument(
        "--dataset-ids",
        type=str,
        default="",
        help="Lista de OpenML dataset IDs separados por coma. Ej: 31,37,50"
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="En vez de pasar dataset-ids, busca en OpenML usando criterios simples y toma los primeros N."
    )
    parser.add_argument("--task-type", type=str, default="Supervised Classification")
    parser.add_argument("--n-samples-min", type=int, default=500)
    parser.add_argument("--n-samples-max", type=int, default=5000)
    parser.add_argument("--n-features-max", type=int, default=50)
    parser.add_argument("--take-n", type=int, default=20, help="Cuántos datasets tomar del search")
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache de OpenML (opcional)")
    parser.add_argument("--out-dir", type=str, default=None, help="Override del store_dir (opcional)")
    parser.add_argument("--cv-folds", type=int, default=None, help="Override de cv_folds (opcional)")

    args = parser.parse_args()

    cfg = SelectorV1Config()
    store_dir = args.out_dir or cfg.store_dir
    cv_folds = args.cv_folds or cfg.cv_folds

    os.makedirs(store_dir, exist_ok=True)
    db = MetaDatabase(store_dir)

    # 1) Definir datasets
    dataset_ids: List[int] = []
    if args.search:
        print("[INFO] Buscando datasets en OpenML...")
        df = search_openml_datasets(
            task_type=args.task_type,
            n_samples_min=args.n_samples_min,
            n_samples_max=args.n_samples_max,
            n_features_max=args.n_features_max,
        )
        # En OpenML tasks list, el dataset_id suele estar en columna 'dataset_id'
        if "dataset_id" not in df.columns:
            raise RuntimeError("No se encontró columna 'dataset_id' en el resultado de search_openml_datasets()")

        dataset_ids = df["dataset_id"].astype(int).head(args.take_n).tolist()
        print(f"[INFO] Dataset IDs seleccionados (N={len(dataset_ids)}): {dataset_ids}")
    else:
        dataset_ids = parse_dataset_ids(args.dataset_ids)

    if not dataset_ids:
        # Defaults pequeños y razonables, por si no pasan nada (clasificación en OpenML).
        dataset_ids = [31, 37, 50, 54, 61, 179, 181, 188] 
        print(f"[INFO] No se pasaron datasets; usando defaults: {dataset_ids}")

    # 2) Cargar datasets
    datasets = load_datasets_openml(dataset_ids, cache_dir=args.cache_dir)
    if len(datasets) < 3:
        raise RuntimeError("Muy pocos datasets cargados. Necesitas al menos 3 para meta-learning razonable.")

    # 3) Meta-features (vector estable)
    schema = MetaFeatureSchema.default()
    meta_features_df = compute_meta_features_table(datasets, schema=schema)

    # 4) Ground truth performances (CV por algoritmo) + configs base
    benchmarker = GroundTruthBenchmarker(
        algorithms=list(cfg.algorithms),
        base_hyperparams=cfg.base_hyperparams,
        cv_folds=cv_folds,
        random_state=cfg.random_state,
        metric=cfg.metric,
    )
    performances_df, configs_df = benchmarker.build_tables(datasets, meta_features_df)

    # 5) Guardar meta-base
    db.save(meta_features_df, performances_df, configs_df=configs_df)

    print(f"\n[OK] Meta-base construida en: {store_dir}")
    print(f" - Meta-features: {db.meta_features_path}")
    print(f" - Performances:  {db.performances_path}")
    print(f" - Configs base:  {db.configs_path}")
    print("\nSiguiente paso:")
    print("  python variants/selector_v1/train_meta_models.py")


if __name__ == "__main__":
    main()
