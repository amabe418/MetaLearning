import os
import pandas as pd

def load_features(metafeature, path, kind):
    filename = f"{kind}_representations.csv"
    df = pd.read_csv(os.path.join(path, filename))
    return df[["task_id"] + metafeature.basic_columns.split(",")]

def load_basic_features(metafeature, path):
    return load_features(metafeature, path, "basic")

def load_bootstrap_features(metafeature, path):
    return load_features(metafeature, path, "bootstrap")

def load_target_features(pipeline, path):
    """
    Carga las representaciones objetivo (target representations) de un pipeline/algoritmo.
    
    Args:
        pipeline: Objeto con atributo 'name' que indica el nombre del algoritmo (ej: 'adaboost', 'random_forest', 'svm')
        path: Directorio base donde están los archivos (normalmente 'data/')
    
    Returns:
        pd.DataFrame con las representaciones objetivo
    """
    # Los archivos están directamente en data/, no en subdirectorio
    filename = pipeline.name + "_target_representation.csv"
    filepath = os.path.join(path, filename)
    return pd.read_csv(filepath)
    # Intentar múltiples rutas posibles para compatibilidad
    possible_paths = [
        os.path.join(path, pipeline.name + "_target_representations.csv"),  # Ruta estándar
        os.path.join(path, "top_preprocessed_target_representation", pipeline.name + "_target_representation.csv"),  # Ruta original
    ]
    
    for file_path in possible_paths:
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
    
    raise FileNotFoundError(f"No se encontró target representation para {pipeline.name} en ninguna de las rutas: {possible_paths}")

def load_raw_target_features(pipeline, path):
    """
    Carga las representaciones objetivo raw (si existen en algún lugar específico).
    Por ahora, usa la misma ubicación que load_target_features.
    """
    filename = pipeline + "_target_representation.csv"
    filepath = os.path.join(path, filename)
    df_hp = pd.read_csv(filepath)
    if "predictive_accuracy" in df_hp.columns:
        df_hp = df_hp.drop(["predictive_accuracy"], axis=1)
    return df_hp