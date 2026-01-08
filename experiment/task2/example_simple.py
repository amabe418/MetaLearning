"""
Simple Example: Task 2 Refactored Implementation
Demonstrates basic usage with minimal configuration
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from experiment.task2.task2_refactored import (
    get_basic_representations,
    get_raw_target_representations,
    run_task2
)


def create_sample_data():
    """
    Create sample data for testing (if real data not available).
    """
    print("Creating sample data for demonstration...")
    
    # Sample meta-features (5 tasks, 10 features)
    np.random.seed(42)
    n_tasks = 5
    n_features = 10
    
    task_ids = [3, 6, 11, 12, 14]
    features = np.random.randn(n_tasks, n_features)
    
    metafeatures = pd.DataFrame(
        features,
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    metafeatures.insert(0, 'task_id', task_ids)
    
    # Sample raw hyperparameters (20 configs per task)
    configs_per_task = 20
    all_configs = []
    
    for task_id in task_ids:
        for _ in range(configs_per_task):
            config = {
                'task_id': task_id,
                'n_estimators': np.random.choice([50, 100, 200, 500]),
                'learning_rate': np.random.choice([0.001, 0.01, 0.1, 1.0]),
                'max_depth': np.random.choice([1, 2, 3, 5, 10]),
            }
            all_configs.append(config)
    
    raw_hps = pd.DataFrame(all_configs)
    
    return metafeatures, raw_hps


def main():
    
    # Try to load real data, fall back to sample data
    data_path = PROJECT_ROOT / "data"
    
    try:
        print("Attempting to load real data...")
        
        # Try basic representations
        metafeatures = get_basic_representations(str(data_path))
        print(f"✓ Loaded basic meta-features: {metafeatures.shape}")
        
        # Try raw hyperparameters for adaboost
        raw_hps = get_raw_target_representations(str(data_path), "adaboost")
        print(f"✓ Loaded raw hyperparameters: {raw_hps.shape}")
        
        use_real_data = True
        
    except FileNotFoundError as e:
        print(f"⚠ Real data not found: {e}")
        print("Using sample data instead...\n")
        
        metafeatures, raw_hps = create_sample_data()
        use_real_data = False
    
    # Display data info
    print("\n" + "="*70)
    print("DATA SUMMARY")
    print("="*70)
    print(f"Meta-features shape: {metafeatures.shape}")
    print(f"  Tasks: {len(metafeatures)}")
    print(f"  Features: {metafeatures.shape[1] - 1}")
    print(f"\nRaw hyperparameters shape: {raw_hps.shape}")
    print(f"  Total configs: {len(raw_hps)}")
    print(f"  Hyperparameters: {raw_hps.shape[1] - 1}")
    print(f"  Configs per task: {len(raw_hps) / len(metafeatures):.1f} (avg)")
    
    # Select test task
    test_task_id = int(metafeatures['task_id'].iloc[0])
    
    print(f"\n" + "="*70)
    print(f"RUNNING TASK 2")
    print("="*70)
    print(f"Test task ID: {test_task_id}")
    print(f"k neighbors: 3")
    print(f"Iterations: 5")
    
    # Run Task 2
    results = run_task2(
        metafeatures=metafeatures,
        raw_target_representations=raw_hps,
        test_task_id=test_task_id,
        k=3,
        nb_iterations=5,
        seed=42
    )
    
    # Display results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    print(f"\nTest Task: {results['test_task_id']}")
    
    print(f"\nNearest Neighbors (k={results['k']}):")
    for i, (nid, dist) in enumerate(zip(results['neighbor_ids'], results['neighbor_distances']), 1):
        print(f"  {i}. Task {nid:>5} - Distance: {dist:.6f} - Weight: {np.exp(-i+1):.6f}")
    
    print(f"\nSampled Configurations ({results['nb_iterations']} total):")
    print(results['sampled_configs'])
    
    print(f"\nConfiguration Statistics:")
    print(results['sampled_configs'].describe())
    
    # Save results if using real data
    if use_real_data:
        output_dir = PROJECT_ROOT / "experiment" / "task2" / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"example_task{test_task_id}.csv"
        results['sampled_configs'].to_csv(output_file, index=False)
        print(f"\n✓ Sampled configurations saved to: {output_file}")
    
    print("\n" + "="*70)
    print("EXAMPLE COMPLETED SUCCESSFULLY!")
    print("="*70)
    


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
