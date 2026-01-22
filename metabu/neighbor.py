import os
from omegaconf import OmegaConf
from experiments.tasks import return_neighbors



def load_config(config_base_path="./conf",
                pipeline_name = "adaboost",
                metafeature_name = "metabu",
                task_name = "neighbor"):


    # Config general
    config_yaml = os.path.join(config_base_path, "config.yaml")
    cfg_main = OmegaConf.load(config_yaml)

    # Pipeline
    pipeline_yaml = os.path.join(config_base_path, "pipeline", f"{pipeline_name}.yaml")
    cfg_pipeline = OmegaConf.load(pipeline_yaml)

    # Metafeature
    metafeature_yaml = os.path.join(config_base_path, "metafeature", f"{metafeature_name}.yaml")
    cfg_metafeature = OmegaConf.load(metafeature_yaml)

    # Task
    task_yaml = os.path.join(config_base_path, "task", f"{task_name}.yaml")
    cfg_task = OmegaConf.load(task_yaml)

    return cfg_main, cfg_pipeline, cfg_metafeature, cfg_task


def compute(dataset, ndcg = 10,meta="metabu", algorithm="adaboost"):


    cfg_main, cfg_pipeline, cfg_metafeature, cfg_task = load_config(pipeline_name=algorithm,
                                                                    metafeature_name=meta,
                                                                   task_name="neighbor")
    print(cfg_main, cfg_pipeline, cfg_metafeature, cfg_task)

    cfg = OmegaConf.create({
        "seed": cfg_main.get("seed", 42),
        "pipeline": cfg_pipeline,
        "metafeature": cfg_metafeature,
        "task": cfg_task,
        "openml_tid": cfg_main.get("openml_tid", dataset),
        "data_path": cfg_main.get("data_path", "./data_metabu_iclr"),
        "output_file": cfg_main.get("output_file", None)
    })

    cfg.task.ndcg = ndcg
    cfg.openml_tid = dataset

    neighbors = return_neighbors(cfg)

    print(f"los {ndcg} vecinos de {dataset} son :\n", neighbors)

    return neighbors


compute(11,63)