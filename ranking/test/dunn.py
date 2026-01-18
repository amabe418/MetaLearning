import numpy as np
import scikit_posthocs as sp

def dunn_test(data, groups, p_adjust='bonferroni'):
    """
    Realiza el test de Dunn para comparaciones múltiples tras un Friedman significativo.
    Args:
        data (array-like): Datos de las observaciones.
        groups (array-like): Etiquetas de grupo para cada observación.
        p_adjust (str): Método de corrección de p-valor ('bonferroni', 'holm', etc.).
    Returns:
        DataFrame: Resultados del test de Dunn con p-valores ajustados.
    """
    return sp.posthoc_dunn(data, groups, p_adjust=p_adjust)

if __name__ == "__main__":
    # Ejemplo de uso real
    # Supongamos 3 algoritmos evaluados en 5 datasets (ranks)
    data = np.array([
        [1, 2, 3],  # dataset 1: rank de cada algoritmo
        [2, 1, 3],  # dataset 2
        [1, 3, 2],  # dataset 3
        [2, 1, 3],  # dataset 4
        [3, 2, 1],  # dataset 5
    ])
    # Convertimos a formato largo
    data_long = data.flatten()
    groups = np.tile(np.array(["alg1", "alg2", "alg3"]), data.shape[0])
    result = dunn_test(data_long, groups)
    print("Resultados del test de Dunn:")
    print(result)
