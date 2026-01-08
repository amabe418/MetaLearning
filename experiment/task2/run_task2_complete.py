"""
Complete Task 2 Pipeline - Refactored Implementation

This script provides an end-to-end pipeline for Task 2:
1. Load METABU meta-features
2. Load raw hyperparameter configurations
3. Find nearest neighbors for a test task
4. Sample configurations with exponential weighting
5. Evaluate configurations using SMAC

"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from ConfigSpace.read_and_write import json as cs_json

# Import our refactored modules
from task2_refactored import (
    get_basic_representations,
    get_metabu_representations,
    get_raw_target_representations,
    run_task2,
    run_task2_leave_one_out
)

# SMAC evaluator (optional)
try:
    from smac_evaluator import evaluate_task2_results
    SMAC_AVAILABLE = True
except ImportError:
    SMAC_AVAILABLE = False


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG = {
    # Data paths
    'data_path': str(PROJECT_ROOT / "data"),
    
    # Pipeline to use
    'pipeline': 'adaboost',  # Options: 'adaboost', 'random_forest', 'libsvm_svc'
    
    # Meta-feature type
    'metafeature_type': 'metabu',  # Options: 'metabu', 'basic'
    
    # Task 2 parameters
    'k': 5,                    # Number of nearest neighbors
    'nb_iterations': 10,       # Number of configurations to sample
    'seed': 42,                # Random seed
    
    # Execution mode
    'mode': 'single',          # Options: 'single', 'leave_one_out'
    'test_task_id': None,      # For 'single' mode (None = use first available)
    
    # Evaluation
    'evaluate_with_smac': False,  # Set to True to actually evaluate configs
    
    # Output
    'output_dir': str(PROJECT_ROOT / "experiment" / "task2" / "results"),
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_config_space(pipeline_name, data_path):
    """
    Load ConfigSpace for a pipeline.
    
    Args:
        pipeline_name: Name of the pipeline
        data_path: Path to data directory
    
    Returns:
        ConfigSpace object
    """
    configspace_file = os.path.join(
        data_path, 
        "configspace", 
        f"{pipeline_name}_configspace.json"
    )
    
    if not os.path.exists(configspace_file):
        raise FileNotFoundError(
            f"ConfigSpace file not found: {configspace_file}\n"
            f"Available pipelines: check {os.path.join(data_path, 'configspace')}"
        )
    
    with open(configspace_file, 'r') as fh:
        json_string = fh.read()
        return cs_json.read(json_string)


def save_results(results, output_dir, filename):
    """
    Save results to JSON file.
    
    Args:
        results: Results dict or list
        output_dir: Output directory
        filename: Output filename
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / filename
    
    # Convert DataFrames to dicts for JSON serialization
    if isinstance(results, dict):
        results_serializable = {}
        for key, value in results.items():
            if isinstance(value, pd.DataFrame):
                results_serializable[key] = value.to_dict(orient='records')
            elif isinstance(value, np.ndarray):
                results_serializable[key] = value.tolist()
            else:
                results_serializable[key] = value
    elif isinstance(results, list):
        results_serializable = []
        for item in results:
            if isinstance(item, dict):
                item_serializable = {}
                for key, value in item.items():
                    if isinstance(value, pd.DataFrame):
                        item_serializable[key] = value.to_dict(orient='records')
                    elif isinstance(value, np.ndarray):
                        item_serializable[key] = value.tolist()
                    else:
                        item_serializable[key] = value
                results_serializable.append(item_serializable)
            else:
                results_serializable.append(item)
    else:
        results_serializable = results
    
    with open(output_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")


# ============================================================================
# MAIN EXECUTION FUNCTIONS
# ============================================================================

def run_single_task(config):
    """
    Run Task 2 for a single test task.
    
    Args:
        config: Configuration dict
    
    Returns:
        Results dict
    """
    print("\n" + "="*70)
    print("MODE: Single Task Evaluation")
    print("="*70)
    
    # Load meta-features
    if config['metafeature_type'] == 'metabu':
        print(f"\nLoading METABU meta-features for {config['pipeline']}...")
        metafeatures = get_metabu_representations(
            config['data_path'], 
            config['pipeline']
        )
    else:
        print(f"\nLoading basic meta-features...")
        metafeatures = get_basic_representations(config['data_path'])
    
    print(f"  Loaded {len(metafeatures)} tasks")
    print(f"  Meta-feature dimensions: {metafeatures.shape[1] - 1}")  # -1 for task_id
    
    # Load raw hyperparameter configurations
    print(f"\nLoading raw hyperparameter configurations...")
    raw_hps = get_raw_target_representations(
        config['data_path'], 
        config['pipeline']
    )
    print(f"  Loaded {len(raw_hps)} configurations")
    print(f"  Hyperparameter dimensions: {raw_hps.shape[1] - 1}")  # -1 for task_id
    
    # Determine test task
    test_task_id = config['test_task_id']
    if test_task_id is None:
        test_task_id = int(metafeatures['task_id'].iloc[0])
        print(f"\nNo test_task_id specified, using: {test_task_id}")
    
    # Run Task 2
    results = run_task2(
        metafeatures=metafeatures,
        raw_target_representations=raw_hps,
        test_task_id=test_task_id,
        k=config['k'],
        nb_iterations=config['nb_iterations'],
        seed=config['seed']
    )
    
    # Optionally evaluate with SMAC
    if config['evaluate_with_smac']:
        if not SMAC_AVAILABLE:
            print("\n⚠ SMAC not available. Skipping evaluation.")
            print("  Install with: pip install smac")
        else:
            print("\n" + "="*70)
            print("Evaluating configurations with SMAC...")
            print("="*70)
            
            # Load config space
            cs = load_config_space(config['pipeline'], config['data_path'])
            
            # Evaluate
            eval_results = evaluate_task2_results(
                task_id=test_task_id,
                pipeline_name=config['pipeline'],
                config_space=cs,
                sampled_configs=results['sampled_configs'],
                seed=config['seed']
            )
            
            results['evaluation_results'] = eval_results
            
            # Print summary
            print("\nEvaluation Summary:")
            for i, res in enumerate(eval_results, 1):
                print(f"  Config {i}: {res['performance']:.4f} ({res['status']})")
    
    return results


def run_leave_one_out_mode(config):
    """
    Run Task 2 in leave-one-out mode.
    
    Args:
        config: Configuration dict
    
    Returns:
        List of results
    """
    print("\n" + "="*70)
    print("MODE: Leave-One-Out Evaluation")
    print("="*70)
    
    # Load meta-features
    if config['metafeature_type'] == 'metabu':
        print(f"\nLoading METABU meta-features for {config['pipeline']}...")
        metafeatures = get_metabu_representations(
            config['data_path'], 
            config['pipeline']
        )
    else:
        print(f"\nLoading basic meta-features...")
        metafeatures = get_basic_representations(config['data_path'])
    
    print(f"  Loaded {len(metafeatures)} tasks")
    
    # Load raw hyperparameter configurations
    print(f"\nLoading raw hyperparameter configurations...")
    raw_hps = get_raw_target_representations(
        config['data_path'], 
        config['pipeline']
    )
    print(f"  Loaded {len(raw_hps)} configurations")
    
    # Run leave-one-out
    results = run_task2_leave_one_out(
        metafeatures=metafeatures,
        raw_target_representations=raw_hps,
        k=config['k'],
        nb_iterations=config['nb_iterations'],
        seed=config['seed']
    )
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Main execution function.
    """
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║              Task 2 - Complete Pipeline                          ║
    ║              Refactored Implementation                           ║
    ║              Based on /aaaa/tasks.py                             ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    print("Configuration:")
    print(f"  Pipeline:           {CONFIG['pipeline']}")
    print(f"  Meta-feature type:  {CONFIG['metafeature_type']}")
    print(f"  k neighbors:        {CONFIG['k']}")
    print(f"  Iterations:         {CONFIG['nb_iterations']}")
    print(f"  Mode:               {CONFIG['mode']}")
    print(f"  Evaluate w/ SMAC:   {CONFIG['evaluate_with_smac']}")
    
    try:
        # Execute based on mode
        if CONFIG['mode'] == 'single':
            results = run_single_task(CONFIG)
            
            # Save results
            filename = f"task2_single_{CONFIG['pipeline']}_task{results['test_task_id']}.json"
            save_results(results, CONFIG['output_dir'], filename)
            
            print("\n✓ Single task evaluation completed!")
            
        elif CONFIG['mode'] == 'leave_one_out':
            results = run_leave_one_out_mode(CONFIG)
            
            # Save results
            filename = f"task2_loo_{CONFIG['pipeline']}_k{CONFIG['k']}.json"
            save_results(results, CONFIG['output_dir'], filename)
            
            print(f"\n✓ Leave-one-out evaluation completed!")
            print(f"  Processed {len(results)} tasks")
            
        else:
            print(f"\n✗ Unknown mode: {CONFIG['mode']}")
            print(f"  Available modes: 'single', 'leave_one_out'")
            return
        
        print("\n" + "="*70)
        print("Task 2 pipeline completed successfully!")
        print("="*70)
        
    except FileNotFoundError as e:
        print(f"\n✗ File not found error:")
        print(f"  {e}")
        print(f"\nPlease check:")
        print(f"  1. Data path: {CONFIG['data_path']}")
        print(f"  2. Pipeline: {CONFIG['pipeline']}")
        print(f"  3. Required files exist")
        
    except Exception as e:
        print(f"\n✗ Unexpected error:")
        print(f"  {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    
