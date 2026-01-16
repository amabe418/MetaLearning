from multiprocessing import Pool
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from model.distance import distance_wasserstein

def get_cost_matrix(
    target_repr, task_ids, verbose, ncpus=1
):
    """
    Calcula la matriz de distancias entre tareas usando una función de distancia dada.
    
    Args:
        target_repr (pd.DataFrame): DataFrame con las representaciones de los targets de cada tarea.
        task_ids (list[int]): Lista de IDs de las tareas a comparar.
        verbose (bool): Si True, muestra una barra de progreso.
        column_id (str): Nombre de la columna que identifica cada tarea (por ejemplo 'task_id').
        pairwise_target_dist_func (function): Función que recibe un par de arrays y devuelve su distancia.
        ncpus (int, opcional): Número de procesos paralelos a usar. Por defecto 1.
    
    Returns:
        np.ndarray: Matriz cuadrada de tamaño len(task_ids) x len(task_ids) con las distancias normalizadas.
    
    Notas:
        - La matriz se hace simétrica y la diagonal se llena de ceros.
        - La matriz se normaliza dividiendo por su valor máximo.
    """

    print(target_repr)

    matrix_ot_distance = []
    for task_a in tqdm(task_ids, disable=not verbose):
        temp_distance = []
        #p = Pool(ncpus)
        params = [
            (target_repr.loc[target_repr.task_id == task_a].drop(
                ['task_id'], axis=1).values,
             target_repr.loc[target_repr.task_id == task_b].drop(
                 ['task_id'], axis=1).values
             ) for task_b in task_ids
        ]
        temp_distance = [distance_wasserstein(_) for _ in params]

        matrix_ot_distance.append(temp_distance)

    matrix_ot_distance = np.array(matrix_ot_distance)
    np.fill_diagonal(matrix_ot_distance, 0)
    matrix_ot_distance /= matrix_ot_distance.max()
    for i in range(matrix_ot_distance.shape[0]):
        for j in range(i):
            matrix_ot_distance[i, j] = matrix_ot_distance[j, i]

    return matrix_ot_distance


def get_ndcg_score(dist_pred, dist_true, k=10):
    """
    Calcula el NDCG (Normalized Discounted Cumulative Gain) entre un ranking predicho y uno verdadero.
    
    Args:
        dist_pred (np.ndarray): Matriz de distancias predichas (o ranking predicho).
        dist_true (np.ndarray): Matriz de distancias verdaderas (o ranking verdadero).
        k (int, opcional): Número de top elementos a considerar en el cálculo. Por defecto 10.
    
    Returns:
        float: Valor de NDCG entre los rankings predicho y verdadero.
    
    Notas:
        - Convierte las distancias en rankings binarios (top-k = 1, resto = 0).
        - NDCG evalúa qué tan bien coincide el ranking predicho con el ranking verdadero.
    """
    pred_rank = dist_pred.argsort().argsort()
    true_rank = dist_true.argsort().argsort()

    pred_rank[np.where(pred_rank < k)] = 1
    pred_rank[np.where(pred_rank >= k)] = 0
    true_rank[np.where(true_rank < k)] = 1
    true_rank[np.where(true_rank >= k)] = 0

    return ndcg_score(y_true=true_rank, y_score=pred_rank, k=k)


def get_pca_importances(data):
    """
    Calcula la importancia de cada feature usando el primer componente principal (PCA).
    
    Args:
        data (np.ndarray or pd.DataFrame): Matriz de datos de shape (n_samples, n_features)
    
    Returns:
        np.ndarray: Vector de tamaño n_features con la magnitud de los pesos del primer componente principal.
    
    Notas:
        - Escala los datos antes de aplicar PCA (media 0, varianza 1).
        - Se devuelve el valor absoluto de los coeficientes del primer componente.
    """
    data_scaled = StandardScaler().fit_transform(data)
    pca = PCA()
    pca.fit_transform(data_scaled)
    return np.abs(pca.components_[0])
