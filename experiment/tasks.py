import math

import numpy as np
from sklearn.metrics import pairwise_distances

from model.utils import get_cost_matrix, get_ndcg_score

from .pipeline_tuner import PipelineTuner
from .utils import load_basic_representations,load_target_representations, load_raw_target_representations,load_configuration_space
from .reprs import load_model_representations

def run_task1(cfg):

    target_reprs = load_target_representations(pipeline=cfg.pipeline, path=cfg.data_path)

    list_ids = sorted(list(target_reprs["task_id"].unique()))

    if cfg.openml_tid not in list_ids:
        raise Exception(f"OpenML task {cfg.openml_tid} does not have target representations.")

    basic_reprs = load_basic_representations(metafeature=cfg.metafeature, path=cfg.data_path)

    basic_reprs = basic_reprs[basic_reprs.task_id.isin(list_ids)]

    if cfg.metafeature.name == "metafeatx":
        train_ids = [_ for _ in list_ids if _ != cfg.openml_tid]
        test_ids = [cfg.openml_tid]

        basic_reprs = load_model_representations(cfg, basic_reprs, target_reprs, list_ids, train_ids, test_ids)

    basic_reprs = basic_reprs.set_index("task_id")
    print(basic_reprs)

    true_dist = get_cost_matrix(target_repr=target_reprs, task_ids=list_ids, verbose=False)
    print(true_dist.shape)
    print(true_dist)
    pred_dist = pairwise_distances(basic_reprs.loc[list_ids])

    print(pred_dist)
    print(pred_dist.shape)

    id_test = list_ids.index(cfg.openml_tid)

    print("Task 1: \n- pipeline: {0} \n- Metafeature: {1} \n- OpenML task: {3} \n- NDCG@{2}: {4}".format(
        cfg.pipeline.name,
        cfg.metafeature.name,
        cfg.task.ndcg,
        cfg.openml_tid,
        get_ndcg_score(dist_pred=np.array([pred_dist[id_test]]), dist_true=np.array([true_dist[id_test]]),
                       k=cfg.task.ndcg)
    ))

    if cfg.output_file is not None:
        print(12)
        with open(cfg.output_file, 'a') as the_file:
            the_file.write("{0},{1},{2},{3},{4}\n".format(
                cfg.pipeline.name,
                cfg.metafeature.name,
                cfg.openml_tid,
                cfg.task.ndcg,
                get_ndcg_score(dist_pred=np.array([pred_dist[id_test]]), dist_true=np.array([true_dist[id_test]]),
                               k=cfg.task.ndcg)
            ))


def run_task2(cfg):
    target_reprs = load_target_representations(pipeline=cfg.pipeline, path=cfg.data_path)
    list_ids = sorted(list(target_reprs["task_id"].unique()))

    if cfg.openml_tid not in list_ids:
        raise Exception(f"OpenML task {cfg.openml_tid} does not have target representations.")

    basic_reprs = load_basic_representations(metafeature=cfg.metafeature, path=cfg.data_path)
    basic_reprs = basic_reprs[basic_reprs.task_id.isin(list_ids)]

    if cfg.metafeature.name == "metafeatx":
        train_ids = [_ for _ in list_ids if _ != cfg.openml_tid]
        test_ids = [cfg.openml_tid]

        basic_reprs = load_model_representations(cfg, basic_reprs, target_reprs, list_ids, train_ids, test_ids)

    basic_reprs = basic_reprs.set_index("task_id")

    pred_dist = pairwise_distances(basic_reprs.loc[list_ids])
    id_test = list_ids.index(cfg.openml_tid)

    # Get id neighbors
    id_neighbors = [list_ids[_] for _ in pred_dist[id_test].argsort()[1:]][:cfg.task.ndcg]

    print(id_neighbors)

    hps = load_raw_target_representations(pipeline=cfg.pipeline.name, path=cfg.data_path)
    hps = hps[hps.task_id.isin(id_neighbors)]
    hps["weights"] = hps.task_id.map(lambda x: math.exp(-id_neighbors.index(x)))
    hps["weights"] /= hps["weights"].sum()

    idx = np.random.choice(hps.index, cfg.task.nb_iterations, p=hps.weights, replace=False)
    cs = load_configuration_space(cfg)

    run = PipelineTuner(pipeline=cfg.pipeline.name, config_space=cs, seed=cfg.seed)
    results = run.exec(task_id=cfg.openml_tid, hps=hps.drop(["task_id", "weights"], axis=1).loc[idx].to_dict(orient="records"), counter=0)

    print("Results with task_id:{}, meta-feature={}, and pipeline={}".format(results[0]["task_id"],
                                                                             cfg.pipeline.name,
                                                                             results[0]["pipeline"]))
    for res in results:
        print("Iter={}\n\t hp={}\n\t perf={}".format(res["hp_id"] + 1, res["hp"], res["performance"]))


