import numpy as np
from utils import load_basic_features, load_target_features
from reprs import get_model_representations
from model.utils import get_cost_matrix, get_ndcg_score
from sklearn.metrics import pairwise_distances


def run_task1(cfg):
    target_reprs = load_target_features(pipeline=cfg.pipeline, path=cfg.data_path)
    list_ids = sorted(list(target_reprs["task_id"].unique()))

    if cfg.openml_tid not in list_ids:
        raise Exception(f"OpenML task {cfg.openml_tid} does not have target representations.")

    basic_reprs = load_basic_features(metafeature=cfg.metafeature, path=cfg.data_path)
    basic_reprs = basic_reprs[basic_reprs.task_id.isin(list_ids)]

    if cfg.metafeature.name == "model_metafeatx":
        train_ids = [_ for _ in list_ids if _ != cfg.openml_tid]
        test_ids = [cfg.openml_tid]

        basic_reprs = get_model_representations(cfg, basic_reprs, target_reprs, list_ids, train_ids, test_ids)

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

