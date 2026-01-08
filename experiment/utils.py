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
    return pd.read_csv(
        os.path.join(path, "top_preprocessed_target_representation", pipeline.name + "_target_representation.csv"))

def load_raw_target_features(pipeline, path):
    df_hp= pd.read_csv(
        os.path.join(path, "top_raw_target_representation", pipeline + "_target_representation.csv")).drop(["predictive_accuracy"], axis=1)
    return df_hp