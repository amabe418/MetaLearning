import numpy as np
import pandas as pd
from model.metafeatx import MetaFeatX
from utils import load_bootstrap_features

def get_model_representations(cfg, basic_reprs, target_reprs, list_ids, train_ids, test_ids):
    """
    Entrena el modelo sobre un conjunto de tareas y devuelve las representaciones meta aprendidas.

    Args:
        cfg: configuración que contiene metafeature, data_path, task.ndcg y seed.
        basic_reprs: pd.DataFrame con las representaciones básicas de todas las tareas.
        target_reprs: pd.DataFrame con las representaciones objetivo.
        list_ids: lista de task_id que se van a considerar.
        train_ids: lista de task_id de entrenamiento.
        test_ids: lista de task_id de test.

    Returns:
        pd.DataFrame con las representaciones aprendidas por el modelo (columnas col0..coln + task_id).
    """
    # =========================
    # Marcar dataset original y bootstrap
    # =========================
    basic_reprs["bootstrap"] = 0
    bootstrap_reprs = load_bootstrap_features(metafeature=cfg.metafeature, path=cfg.data_path)
    bootstrap_reprs = bootstrap_reprs[bootstrap_reprs.task_id.isin(list_ids)]
    bootstrap_reprs["bootstrap"] = 1

    # =========================
    # Combinar datos originales + bootstrap
    # =========================
    combined_basic_reprs = pd.concat([basic_reprs, bootstrap_reprs], axis=0)
    combined_basic_reprs = pd.concat([
        combined_basic_reprs[combined_basic_reprs.task_id.isin(train_ids)],
        combined_basic_reprs[combined_basic_reprs.task_id.isin(test_ids)]
    ], axis=0)

    # =========================
    # Entrenamiento de MetaFeatX
    # =========================
    model = MetaFeatX(
        alpha=0.5,
        lambda_reg=1e-3,
        learning_rate=0.01,
        early_stopping_patience=20,
        early_stopping_criterion_ndcg=cfg.task.ndcg,
        verbose=False,
        seed=cfg.seed
    )
    repr_train, repr_test = model.train_and_predict(
        basic_reprs=combined_basic_reprs.drop(["bootstrap"], axis=1),
        target_reprs=target_reprs,
        column_id="task_id",
        train_ids=train_ids,
        test_ids=test_ids
    )

    # =========================
    # Combinar resultados y devolver
    # =========================
    model_reprs = np.concatenate([repr_train, repr_test], axis=0)
    model_reprs = pd.DataFrame(model_reprs, columns=[f"col{_}" for _ in range(model_reprs.shape[1])])
    model_reprs["task_id"] = combined_basic_reprs["task_id"].values
    model_reprs["bootstrap"] = combined_basic_reprs["bootstrap"].values

    # Solo devolver las filas originales (no bootstrap)
    return model_reprs[model_reprs.bootstrap == 0].drop(["bootstrap"], axis=1)
