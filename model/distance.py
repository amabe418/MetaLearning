import ot
import numpy as np
from sklearn.linear_model import LinearRegression
import math


def distance_wasserstein(distributions, return_map_matrix=False):
    """
    Calcula la distancia de Wasserstein (EMD) entre dos distribuciones.

    Args:
        distributions (tuple of np.ndarray): Tupla de dos arrays (distribution_a, distribution_b) 
            representando las distribuciones a comparar.
        return_map_matrix (bool, opcional): Si True, devuelve también la matriz de transporte óptimo. 
            Por defecto False.

    Returns:
        float o tuple: 
            - Si return_map_matrix=False: devuelve la distancia de Wasserstein (float)
            - Si return_map_matrix=True: devuelve (distancia, matriz_G0) donde matriz_G0 es la matriz de transporte óptimo.

    Notas:
        - La matriz de costes se normaliza dividiendo por su valor máximo.
        - Se asume que las distribuciones son uniformes (cada punto tiene peso igual).
        - Utiliza la implementación de EMD de la librería POT (Python Optimal Transport).
    """

    # Desempaquetar las dos distribuciones a comparar
    distribution_a, distribution_b = distributions

    # Construir la matriz de costos M, donde M[i, j] es la distancia
    # entre el punto i de la distribución A y el punto j de la distribución B
    M = ot.dist(distribution_a, distribution_b)

    # Normalizar la matriz de costos para que sus valores estén en [0, 1]
    # Esto mejora la estabilidad numérica y evita escalas dominantes
    M /= M.max()

    # Número de puntos en cada distribución
    ns = len(distribution_a)
    nt = len(distribution_b)

    # Definir distribuciones de probabilidad uniformes sobre los puntos
    # Cada punto tiene la misma masa
    a, b = np.ones((ns,)) / ns, np.ones((nt,)) / nt

    # Resolver el problema de transporte óptimo (Earth Mover's Distance)
    # G0 es la matriz de transporte óptimo que indica cuánta masa se
    # transporta de cada punto de A a cada punto de B
    G0 = ot.emd(a, b, M)

    # Calcular la distancia de Wasserstein como el costo total del
    # transporte óptimo: sum_{i,j} G0[i,j] * M[i,j]    
    if return_map_matrix:
        return (G0 * M).sum(), G0
    

    # Devolver también la matriz de transporte si se solicita
    return (G0 * M).sum()


def intrinsic_estimator(matrix_distance):
    """
    Estima la dimensión intrínseca del espacio de tareas a partir de 
    una matriz de distancias entre sus representaciones objetivo.

    La dimensión intrínseca indica cuántos grados de libertad reales
    existen en el espacio de tareas, es decir, cuántas dimensiones latentes
    son suficientes para capturar la variabilidad entre tareas.

    Args:
        matrix_distance (np.ndarray): Matriz cuadrada (N x N) de distancias 
                                      entre tareas, normalmente normalizada.

    Returns:
        int: Dimensión intrínseca estimada, redondeada hacia arriba.
    """
    muL = []
    N = len(matrix_distance)
    Femp = []

    #itera por cada task de la matriz
    for i in range(len(matrix_distance)):
        distances_ = np.unique(matrix_distance[i])

        # toma los dos vecinos mas cercanos
        NN = np.argsort(distances_)[1:3]
        first = NN[0]
        second = NN[1]

        # calcula ratio (invariante a escala y tiene distribucion conocida segun la dimension)
        mu_i = distances_[second] / (distances_[first] + (10 ** (-3)))
        muL.append(mu_i)

    # limpiar outliers (elimina el 10% mas grande)
    muL = np.sort(muL)
    cutoff = int(np.floor(0.9 * len(muL)))
    muL = muL[0 : cutoff + 1]

    # evitar valor invalidos(para el log)
    muL = [x if x > 0 else 1 + 10 ** (-3) for x in muL]

    # transformacion logaritmica
    muL = np.asarray([math.log(mu_i) for mu_i in muL]).reshape(-1, 1)
    
    # construccion de al cdf empirica
    step = 1 / N
    Femp = [i * step for i in range(1, len(muL) + 1)]
    Femp = np.asarray([-math.log(1 - x) for x in Femp]).reshape(-1, 1)

    # Regresion lineal (sin intercepto) para estimar la dimension intrinseca
    clf = LinearRegression(fit_intercept=False)
    clf.fit(muL, Femp)

    # extraer la dimension y redondea
    intrinsic = clf.coef_[0][0]
    return math.ceil(intrinsic)