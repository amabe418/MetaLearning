"""
Task 2 Refactored: Configuration Recommendation using Nearest Neighbors

This implementation:
1. Uses METABU meta-features to find k nearest neighbors
2. Extracts raw hyperparameter configurations from neighbors
3. Weights configurations exponentially by neighbor distance
4. Evaluates sampled configurations using SMAC wrapper
"""

import math
import os
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from pathlib import Path


def get_basic_representations(data_path, metafeature_columns=None):
    """
    Load basic meta-feature representations.
    
    Args:
        data_path: Path to data directory
        metafeature_columns: List of column names to use (if None, use all except task_id)
    
    Returns:
        DataFrame with task_id and meta-features
    """
    basic_reprs = pd.read_csv(os.path.join(data_path, "basic_representations.csv"))
    
    if metafeature_columns is not None:
        return basic_reprs[["task_id"] + metafeature_columns]
    
    return basic_reprs


def get_metabu_representations(data_path, pipeline_name):
    """
    Load METABU learned meta-feature representations.
    
    Args:
        data_path: Path to data directory
        pipeline_name: Name of the pipeline (e.g., 'adaboost', 'random_forest')
    
    Returns:
        DataFrame with task_id and learned meta-features
    """
    file_path = os.path.join(data_path, f"metabu_representations_{pipeline_name}.csv")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"METABU representations not found: {file_path}\n"
            f"You need to train METABU first to generate learned meta-features."
        )
    
    return pd.read_csv(file_path)


def get_raw_target_representations(data_path, pipeline_name):
    """
    Load raw hyperparameter configurations (target representations).
    
    These are the actual hyperparameter values evaluated on each task,
    NOT preprocessed embeddings.
    
    Args:
        data_path: Path to data directory
        pipeline_name: Name of the pipeline
    
    Returns:
        DataFrame with task_id and hyperparameter columns
    """
    file_path = os.path.join(
        data_path, 
        "top_raw_target_representation", 
        f"{pipeline_name}_target_representation.csv"
    )
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Raw target representations not found: {file_path}\n"
            f"Expected format: task_id, hp1, hp2, ..., hpN, predictive_accuracy"
        )
    
    df = pd.read_csv(file_path)
    
    # Remove predictive_accuracy if present (we only want hyperparameters)
    if "predictive_accuracy" in df.columns:
        df = df.drop(["predictive_accuracy"], axis=1)
    
    return df


def get_nearest_neighbors_with_distances(metafeatures, test_task_id, k):
    """
    Find k nearest neighbors using meta-features and return their distances.
    
    Args:
        metafeatures: DataFrame with task_id and meta-features
        test_task_id: ID of the test task
        k: Number of neighbors to find
    
    Returns:
        tuple: (neighbor_task_ids, distances)
            - neighbor_task_ids: List of k nearest neighbor task IDs
            - distances: List of distances to each neighbor
    """
    # Ensure task_id is in the dataframe
    if "task_id" not in metafeatures.columns:
        raise ValueError("metafeatures must have a 'task_id' column")
    
    # Get list of all task IDs
    list_ids = sorted(list(metafeatures["task_id"].unique()))
    
    if test_task_id not in list_ids:
        raise ValueError(f"Test task {test_task_id} not found in metafeatures")
    
    # Set task_id as index
    metafeatures_indexed = metafeatures.set_index("task_id")
    
    # Calculate pairwise distances
    pred_dist = pairwise_distances(metafeatures_indexed.loc[list_ids])
    
    # Get index of test task
    id_test = list_ids.index(test_task_id)
    
    # Get distances from test task to all others
    test_distances = pred_dist[id_test]
    
    # Sort by distance (excluding the test task itself)
    sorted_indices = test_distances.argsort()
    neighbor_indices = [idx for idx in sorted_indices if idx != id_test][:k]
    
    # Get neighbor task IDs and their distances
    neighbor_task_ids = [list_ids[idx] for idx in neighbor_indices]
    neighbor_distances = [test_distances[idx] for idx in neighbor_indices]
    
    return neighbor_task_ids, neighbor_distances


def sample_configurations_with_weights(raw_hps, neighbor_task_ids, nb_iterations):
    """
    Sample configurations from neighbors with exponential weighting.
    
    Configurations from closer neighbors have higher probability of being selected.
    Weight for neighbor i: exp(-i) where i is the rank (0-indexed)
    
    Args:
        raw_hps: DataFrame with task_id and hyperparameter columns
        neighbor_task_ids: List of neighbor task IDs (ordered by distance, closest first)
        nb_iterations: Number of configurations to sample
    
    Returns:
        DataFrame with sampled hyperparameter configurations
    """
    # Filter configurations from neighbor tasks
    hps = raw_hps[raw_hps.task_id.isin(neighbor_task_ids)].copy()
    
    if len(hps) == 0:
        raise ValueError("No configurations found for the specified neighbors")
    
    # Assign exponential weights based on neighbor rank
    # neighbor_task_ids[0] is closest, so it gets highest weight
    def get_weight(task_id):
        if task_id in neighbor_task_ids:
            rank = neighbor_task_ids.index(task_id)
            return math.exp(-rank)
        return 0.0
    
    hps["weights"] = hps.task_id.map(get_weight)
    
    # Normalize weights to sum to 1
    hps["weights"] /= hps["weights"].sum()
    
    # Sample configurations
    if len(hps) < nb_iterations:
        print(f"Warning: Only {len(hps)} configurations available, but {nb_iterations} requested.")
        print(f"Sampling with replacement.")
        replace = True
    else:
        replace = False
    
    idx = np.random.choice(hps.index, size=nb_iterations, p=hps.weights, replace=replace)
    
    # Return sampled configurations (without task_id and weights)
    sampled_configs = hps.drop(["task_id", "weights"], axis=1).loc[idx].reset_index(drop=True)
    
    return sampled_configs


def run_task2(
    metafeatures,
    raw_target_representations,
    test_task_id,
    k=5,
    nb_iterations=10,
    seed=42
):
    """
    Run Task 2: Configuration recommendation using nearest neighbors.
    
    This follows the approach from /aaaa/tasks.py:
    1. Find k nearest neighbors using meta-features
    2. Extract hyperparameter configurations from neighbors
    3. Weight configurations exponentially by neighbor distance
    4. Sample nb_iterations configurations
    5. Return sampled configurations for evaluation
    
    Args:
        metafeatures: DataFrame with task_id and meta-features (can be METABU or basic)
        raw_target_representations: DataFrame with task_id and hyperparameters
        test_task_id: ID of the test task
        k: Number of nearest neighbors (default: 5)
        nb_iterations: Number of configurations to sample (default: 10)
        seed: Random seed for reproducibility
    
    Returns:
        dict with:
            - 'test_task_id': ID of test task
            - 'neighbor_ids': List of k nearest neighbor IDs
            - 'neighbor_distances': Distances to each neighbor
            - 'sampled_configs': DataFrame with sampled configurations
            - 'k': Number of neighbors used
            - 'nb_iterations': Number of configurations sampled
    """
    np.random.seed(seed)
    
    print(f"\n{'='*70}")
    print(f"Task 2: Configuration Recommendation via Nearest Neighbors")
    print(f"{'='*70}")
    print(f"Test task:        {test_task_id}")
    print(f"k neighbors:      {k}")
    print(f"Iterations:       {nb_iterations}")
    print(f"Random seed:      {seed}")
    
    # Step 1: Find k nearest neighbors
    print(f"\nStep 1: Finding {k} nearest neighbors...")
    neighbor_ids, neighbor_distances = get_nearest_neighbors_with_distances(
        metafeatures, test_task_id, k
    )
    
    print(f"  Nearest neighbors:")
    for i, (nid, dist) in enumerate(zip(neighbor_ids, neighbor_distances)):
        print(f"    {i+1}. Task {nid} (distance: {dist:.4f})")
    
    # Step 2: Sample configurations with exponential weighting
    print(f"\nStep 2: Sampling {nb_iterations} configurations from neighbors...")
    sampled_configs = sample_configurations_with_weights(
        raw_target_representations,
        neighbor_ids,
        nb_iterations
    )
    
    print(f"  Sampled {len(sampled_configs)} configurations")
    print(f"  Configuration shape: {sampled_configs.shape}")
    
    # Display sample of configurations
    print(f"\nSample of configurations:")
    print(sampled_configs.head(3))
    
    print(f"\n{'='*70}")
    print(f"Task 2 completed successfully!")
    print(f"{'='*70}\n")
    
    return {
        'test_task_id': test_task_id,
        'neighbor_ids': neighbor_ids,
        'neighbor_distances': neighbor_distances,
        'sampled_configs': sampled_configs,
        'k': k,
        'nb_iterations': nb_iterations
    }


def run_task2_leave_one_out(
    metafeatures,
    raw_target_representations,
    k=5,
    nb_iterations=10,
    seed=42
):
    """
    Run Task 2 in leave-one-out mode for all tasks.
    
    Args:
        metafeatures: DataFrame with task_id and meta-features
        raw_target_representations: DataFrame with task_id and hyperparameters
        k: Number of nearest neighbors
        nb_iterations: Number of configurations to sample per task
        seed: Random seed
    
    Returns:
        List of results dictionaries (one per task)
    """
    all_task_ids = sorted(metafeatures['task_id'].unique())
    results = []
    
    print(f"\n{'='*70}")
    print(f"Task 2: Leave-One-Out Evaluation")
    print(f"{'='*70}")
    print(f"Total tasks:      {len(all_task_ids)}")
    print(f"k neighbors:      {k}")
    print(f"Iterations/task:  {nb_iterations}")
    
    for i, test_task_id in enumerate(all_task_ids, 1):
        print(f"\n[{i}/{len(all_task_ids)}] Processing task {test_task_id}...")
        
        try:
            result = run_task2(
                metafeatures,
                raw_target_representations,
                test_task_id,
                k,
                nb_iterations,
                seed
            )
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    print(f"\n{'='*70}")
    print(f"Leave-One-Out completed: {len(results)}/{len(all_task_ids)} tasks")
    print(f"{'='*70}\n")
    
    return results


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║              Task 2 - Configuration Recommendation               ║
    ║                           ║
    ╚══════════════════════════════════════════════════════════════════╝
    
    This implementation follows the approach from the reference code:
    
    1. Load METABU meta-features (or basic meta-features)
    2. Load raw hyperparameter configurations
    3. For a test task:
       - Find k nearest neighbors using meta-features
       - Extract configurations from neighbors
       - Weight by exp(-rank) where rank is neighbor position
       - Sample configurations according to weights
    4. Return sampled configurations for evaluation
    
    Key differences from previous implementation:
    - Uses RAW hyperparameters, not preprocessed embeddings
    - Exponential weighting by neighbor rank
    - Sampling with replacement if needed
    - Follows exact structure from /aaaa/tasks.py
    
    Usage:
        from task2_refactored import run_task2
        
        results = run_task2(
            metafeatures=metabu_features,
            raw_target_representations=raw_hps,
            test_task_id=3,
            k=5,
            nb_iterations=10
        )
    """)
