import math

import numpy as np
from sklearn.metrics import pairwise_distances

from model import get_cost_matrix, get_ndcg_score

from .pipeline_tuner import PipelineTuner
from .utils import (
    load_basic_representations,
    load_target_representations, 
    load_raw_target_representations,
    load_configuration_space
)

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
    """
    Ejecuta la Tarea 2 del experimento de meta-learning.

    Esta tarea consiste en:
    1. Cargar representaciones objetivo (target representations) de tareas OpenML
       previamente evaluadas con un pipeline dado.
    2. Calcular la similitud entre tareas usando meta-features (básicas o aprendidas).
    3. Seleccionar las tareas más cercanas (vecinas) a la tarea objetivo.
    4. Extraer configuraciones de hiperparámetros usadas en esas tareas vecinas.
    5. Inicializar un optimizador SMAC con dichas configuraciones como punto de partida.
    6. Optimizar el pipeline sobre la tarea objetivo usando SMAC.
    7. Mostrar los resultados obtenidos.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Configuración global del experimento (Hydra), que incluye:
        - cfg.pipeline: pipeline de ML a evaluar
        - cfg.metafeature: tipo de meta-features
        - cfg.data_path: ruta a los datos
        - cfg.openml_tid: ID de la tarea OpenML objetivo
        - cfg.task.ndcg: número de vecinos a considerar
        - cfg.task.nb_iterations: número de configuraciones iniciales
        - cfg.seed: semilla aleatoria
    """
    # 1. Cargar representaciones objetivo (target representations)
    target_reprs = load_target_representations(pipeline=cfg.pipeline, path=cfg.data_path)

    # Obtener la lista de IDs de tareas disponibles
    list_ids = sorted(list(target_reprs["task_id"].unique()))

    # Verificar que la tarea objetivo exista en las representaciones
    if cfg.openml_tid not in list_ids:
        raise Exception(f"OpenML task {cfg.openml_tid} does not have target representations.")
    
    # 2. Cargar representaciones básicas (meta-features)
    basic_reprs = load_basic_representations(metafeature=cfg.metafeature, path=cfg.data_path)

    # Filtrar solo tareas que tengan representaciones objetivo
    basic_reprs = basic_reprs[basic_reprs.task_id.isin(list_ids)]

    # 3. Caso especial: meta-features aprendidas (metafeatx)
    if cfg.metafeature.name == "metafeatx":
        train_ids = [_ for _ in list_ids if _ != cfg.openml_tid]
        test_ids = [cfg.openml_tid]

        # Generar representaciones aprendidas usando un modelo
        basic_reprs = load_model_representations(cfg, basic_reprs, target_reprs, list_ids, train_ids, test_ids)

    basic_reprs = basic_reprs.set_index("task_id")

    # 4. Calcular distancias entre tareas
    pred_dist = pairwise_distances(basic_reprs.loc[list_ids])
    id_test = list_ids.index(cfg.openml_tid)

    # Seleccionar los k vecinos más cercanos (excluyendo la propia tarea)
    id_neighbors = [list_ids[_] for _ in pred_dist[id_test].argsort()[1:]][:cfg.task.ndcg]

    print(id_neighbors)

    # 5. Cargar configuraciones de hiperparámetros de las tareas vecinas
    hps = load_raw_target_representations(pipeline=cfg.pipeline.name, path=cfg.data_path)
    hps = hps[hps.task_id.isin(id_neighbors)]

    # Asignar pesos según cercanía (más cercano ⇒ más peso)
    hps["weights"] = hps.task_id.map(lambda x: math.exp(-id_neighbors.index(x)))
    hps["weights"] /= hps["weights"].sum()

    # Muestrear configuraciones iniciales de forma ponderada
    idx = np.random.choice(hps.index, cfg.task.nb_iterations, p=hps.weights, replace=False)
    
    # 6. Cargar espacio de configuraciones (ConfigSpace)
    cs = load_configuration_space(cfg)

    run = PipelineTuner(pipeline=cfg.pipeline.name, config_space=cs, seed=cfg.seed)

    # Ejecutar SMAC usando las configuraciones iniciales
    results = run.exec(task_id=cfg.openml_tid, hps=hps.drop(["task_id", "weights"], axis=1).loc[idx].to_dict(orient="records"), counter=0)

    print("Results with task_id:{}, meta-feature={}, and pipeline={}".format(results[0]["task_id"],
                                                                             cfg.pipeline.name,
                                                                             results[0]["pipeline"]))
    for res in results:
        print("Iter={}\n\t hp={}\n\t perf={}".format(res["hp_id"] + 1, res["hp"], res["performance"]))


# def run_task3():
#     pass