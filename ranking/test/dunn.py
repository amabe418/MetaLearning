import numpy as np
import scikit_posthocs as sp

def dunn_test(data, p_adjust="bonferroni"):
    """
    Dunn post-hoc test after Friedman.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_datasets, n_algorithms), containing ranks.
    p_adjust : str
        P-value correction method.

    Returns
    -------
    pd.DataFrame
        Pairwise p-values.
    """
    # Cada columna es un grupo (algoritmo)
    groups = [data[:, i] for i in range(data.shape[1])]
    return sp.posthoc_dunn(groups, p_adjust=p_adjust)


if __name__ == "__main__":
    data = np.array([
        [1, 2, 3],
        [2, 1, 3],
        [1, 3, 2],
        [2, 1, 3],
        [3, 2, 1],
    ])

    result = dunn_test(data)
    print("Resultados del test de Dunn:")
    print(result)
