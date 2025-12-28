from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass(frozen=True)
class SelectorV1Config:

    algorithms: List[str] = (
        "RandomForest",
        "SVM",
        "LogisticRegression",
        "KNN",
        "NaiveBayes",
    )

    # Métrica base para el “ground truth”
    metric: str = "accuracy"

    # CV para benchmarks ground truth
    cv_folds: int = 5
    random_state: int = 42

    # Directorio donde guardar tablas (meta_features / performances)
    store_dir: str = "artifacts/selector_v1"

    # Hiperparámetros “base” por algoritmo (configuración mínima viable)
    base_hyperparams: Dict[str, Dict[str, Any]] = None

    def __post_init__(self):
        if self.base_hyperparams is None:
            object.__setattr__(self, "base_hyperparams", {
                "RandomForest": {"n_estimators": 300, "max_depth": None},
                "SVM": {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
                "LogisticRegression": {"C": 1.0, "max_iter": 2000},
                "KNN": {"n_neighbors": 11, "weights": "distance"},
                "NaiveBayes": {},  # GaussianNB sin params críticos
            })
