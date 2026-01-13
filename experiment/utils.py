import pandas as pd
import os
from ConfigSpace.read_and_write import json

def load_basic_representations(metafeature, path):
    basic_representations = pd.read_csv(os.path.join(path, "basic_representations.csv"))
    return basic_representations[["task_id"] + metafeature.basic_columns.split(",")]


def load_bootstrap_representations(metafeature, path):
    basic_representations = pd.read_csv(os.path.join(path, "bootstrap_representations.csv"))
    return basic_representations[["task_id"] + metafeature.basic_columns.split(",")]


def load_target_representations(pipeline, path):
    return pd.read_csv(
        os.path.join(path, "top_preprocessed_target_representation", pipeline.name + "_target_representation.csv"))


def load_raw_target_representations(pipeline, path):
    df_hp= pd.read_csv(
        os.path.join(path, "top_raw_target_representation", pipeline + "_target_representation.csv")).drop(["predictive_accuracy"], axis=1)
    return df_hp


def load_configuration_space(cfg):
    with open(os.path.join(cfg.data_path, "configspace", "{}_configspace.json".format(cfg.pipeline.name)), 'r') as fh:
        json_string = fh.read()
        return json.read(json_string)
