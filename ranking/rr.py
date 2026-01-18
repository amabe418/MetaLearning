import random
import numpy as np
import pandas as pd


def random_ranking_repeated(datasets, algorithms, n_repeats=100, seed=None):
    """
    Generate a stable random ranking by averaging multiple random permutations.

    Parameters
    ----------
    datasets : list
        List of dataset identifiers.
    algorithms : list
        List of algorithm names.
    n_repeats : int
        Number of random repetitions.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Averaged rankings per dataset and algorithm.
    """
    if seed is not None:
        random.seed(seed)

    n_datasets = len(datasets)
    n_algorithms = len(algorithms)

    ranks_accum = np.zeros((n_datasets, n_algorithms))

    for _ in range(n_repeats):
        for i in range(n_datasets):
            perm = algorithms.copy()
            random.shuffle(perm)
            for j, alg in enumerate(algorithms):
                ranks_accum[i, j] += perm.index(alg) + 1

    ranks_accum /= n_repeats

    return pd.DataFrame(
        ranks_accum,
        index=datasets,
        columns=algorithms
    )


if __name__ == "__main__":
    datasets = ["d1", "d2", "d3", "d4"]
    algorithms = ["SVM", "RF", "KNN", "XGB"]

    random_df = random_ranking_repeated(
        datasets,
        algorithms,
        n_repeats=5000,
        seed=42
    )

    print(random_df)