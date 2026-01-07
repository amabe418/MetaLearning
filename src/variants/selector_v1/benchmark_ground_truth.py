from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


@dataclass
class GroundTruthBenchmarker:
    """
    Corre benchmarks para construir performances reales por dataset y algoritmo.
    No modifica tus módulos; produce DataFrames listos para tu meta-learner.
    """
    algorithms: List[str]
    base_hyperparams: Dict[str, Dict[str, Any]]
    cv_folds: int = 5
    random_state: int = 42
    metric: str = "accuracy"

    def _build_estimator(self, alg_name: str):
        hp = self.base_hyperparams.get(alg_name, {})

        if alg_name == "RandomForest":
            model = RandomForestClassifier(
                n_estimators=hp.get("n_estimators", 300),
                max_depth=hp.get("max_depth", None),
                random_state=self.random_state,
                n_jobs=-1
            )
        elif alg_name == "SVM":
            model = SVC(
                kernel=hp.get("kernel", "rbf"),
                C=hp.get("C", 1.0),
                gamma=hp.get("gamma", "scale"),
                probability=False
            )
        elif alg_name == "LogisticRegression":
            model = LogisticRegression(
                C=hp.get("C", 1.0),
                max_iter=hp.get("max_iter", 2000),
                n_jobs=-1
            )
        elif alg_name == "KNN":
            model = KNeighborsClassifier(
                n_neighbors=hp.get("n_neighbors", 11),
                weights=hp.get("weights", "distance"),
            )
        elif alg_name == "NaiveBayes":
            model = GaussianNB()
        else:
            raise ValueError(f"Algoritmo no soportado: {alg_name}")

        return model

    def _build_preprocess(self, X: pd.DataFrame) -> ColumnTransformer:
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in X.columns if c not in num_cols]

        num_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True))
        ])

        cat_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        return ColumnTransformer(
            transformers=[
                ("num", num_pipe, num_cols),
                ("cat", cat_pipe, cat_cols),
            ],
            remainder="drop",
            sparse_threshold=0.0
        )

    def score_dataset(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
        """
        Devuelve:
          - performance dict: alg -> mean_cv_score
          - configs dict: alg -> hyperparams usados (base)
        """
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        preprocess = self._build_preprocess(X)
        performances: Dict[str, float] = {}
        used_configs: Dict[str, Dict[str, Any]] = {}

        for alg in self.algorithms:
            estimator = self._build_estimator(alg)
            pipe = Pipeline(steps=[("prep", preprocess), ("model", estimator)])

            scores = cross_val_score(pipe, X, y, cv=cv, scoring=self.metric, n_jobs=-1)
            performances[alg] = float(np.mean(scores))
            used_configs[alg] = dict(self.base_hyperparams.get(alg, {}))

        return performances, used_configs

    def build_tables(
        self,
        datasets: List[Dict[str, Any]],
        meta_features_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        datasets: lista con {id, X, y}
        meta_features_df: index = dataset_id, cols = meta-features
        """
        perf_rows = []
        cfg_rows = []

        for d in datasets:
            dataset_id = d["id"]
            X = d["X"]
            y = d["y"]

            perf, cfg = self.score_dataset(X, y)

            perf_rows.append({"dataset_id": dataset_id, **perf})

            # configs_df (formato simple, columnas por algoritmo con dict serializado)
            row_cfg = {"dataset_id": dataset_id}
            for alg, params in cfg.items():
                row_cfg[f"{alg}__params"] = str(params)
            cfg_rows.append(row_cfg)

        performances_df = pd.DataFrame(perf_rows).set_index("dataset_id").sort_index()
        configs_df = pd.DataFrame(cfg_rows).set_index("dataset_id").sort_index()

        # Asegurar alineación con meta_features
        performances_df = performances_df.loc[meta_features_df.index]
        configs_df = configs_df.loc[meta_features_df.index]

        return performances_df, configs_df
