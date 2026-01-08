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
    df_hp= pd.read_csv(
        os.path.join(path, "top_raw_target_representation", pipeline + "_target_representation.csv")).drop(["predictive_accuracy"], axis=1)
    return df_hp