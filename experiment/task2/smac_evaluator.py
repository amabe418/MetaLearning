"""
SMAC-based Configuration Evaluator

This module provides functionality to evaluate hyperparameter configurations
using SMAC (Sequential Model-based Algorithm Configuration).
"""

import shutil
from pathlib import Path
import numpy as np
import openml

try:
    from ConfigSpace.configuration_space import Configuration
    from smac.facade.smac_hpo_facade import SMAC4HPO
    from smac.scenario.scenario import Scenario
    from smac.callbacks import IncorporateRunResultCallback
    SMAC_AVAILABLE = True
except ImportError:
    SMAC_AVAILABLE = False
    print("Warning: SMAC not installed. Install with: pip install smac")

# Disable OpenML logging
openml.config.logger.propagate = False
try:
    openml.datasets.dataset.logger.propagate = False
except:
    pass


class ResultCallback(IncorporateRunResultCallback):
    """
    Callback to collect results from SMAC optimization.
    """
    def __init__(self, task_id, pipeline, counter):
        self.task_id = task_id
        self.counter = counter
        self.pipeline = pipeline
        self.results = []

    def __call__(self, smbo, run_info, result, time_left):
        self.results.append({
            "task_id": self.task_id,
            "pipeline": self.pipeline,
            "hp_id": self.counter,
            "hp": run_info.config.get_dictionary(),
            "status": str(result.status),
            "performance": -result.cost  # SMAC minimizes, so negate for accuracy
        })
        self.counter += 1


class ConfigurationEvaluator:
    """
    Evaluates hyperparameter configurations using SMAC.
    
    This class wraps SMAC to evaluate a set of predefined configurations
    on an OpenML task.
    """
    
    def __init__(self, pipeline_name, config_space, seed=42):
        """
        Initialize the evaluator.
        
        Args:
            pipeline_name: Name of the pipeline/classifier
            config_space: ConfigSpace object defining the search space
            seed: Random seed for reproducibility
        """
        if not SMAC_AVAILABLE:
            raise ImportError(
                "SMAC is required for configuration evaluation.\n"
                "Install with: pip install smac"
            )
        
        self.pipeline_name = pipeline_name
        self.config_space = config_space
        self.seed = seed
    
    def generate_black_box_function(self, task_id, n_jobs=1):
        """
        Generate the black-box function for SMAC to optimize.
        
        This function evaluates a configuration on an OpenML task.
        
        Args:
            task_id: OpenML task ID
            n_jobs: Number of parallel jobs
        
        Returns:
            Black-box function that takes a Configuration and returns cost
        """
        def black_box_function(config):
            """
            Evaluate a configuration on the OpenML task.
            
            Args:
                config: SMAC Configuration object
            
            Returns:
                tuple: (cost, additional_info)
                    - cost: negative accuracy (SMAC minimizes)
                    - additional_info: empty dict
            """
            try:
                cfg = config.get_dictionary()
                
                # Import here to avoid circular dependencies
                from aaaa.openml_pimp import set_up_pipeline_for_task
                
                # Set up pipeline with the configuration
                pipe = set_up_pipeline_for_task(task_id, self.pipeline_name)
                pipe.set_params(**cfg)
                
                # Run on OpenML task
                run = openml.runs.run_model_on_task(
                    pipe, 
                    task_id, 
                    avoid_duplicate_runs=False,
                    dataset_format="array", 
                    n_jobs=n_jobs
                )
                
                # Get mean accuracy across folds
                accuracy = np.mean(list(run.fold_evaluations['predictive_accuracy'][0].values()))
                
                # Return negative accuracy (SMAC minimizes)
                return -accuracy, {}
                
            except Exception as e:
                print(f"Error evaluating configuration: {e}")
                # Return worst possible cost
                return 1.0, {}
        
        return black_box_function
    
    def clean_dict(self, dictionary):
        """
        Clean dictionary by converting string booleans to actual booleans.
        
        Args:
            dictionary: Dict with potentially string boolean values
        
        Returns:
            Cleaned dictionary
        """
        c = {"True": True, "False": False}
        return {k: c[v] if v in c else v for k, v in dictionary.items()}
    
    def evaluate_configurations(self, task_id, configurations, counter=0):
        """
        Evaluate a set of configurations on an OpenML task.
        
        Args:
            task_id: OpenML task ID
            configurations: List of dicts, each containing hyperparameter values
            counter: Starting counter for configuration IDs
        
        Returns:
            List of result dicts with keys:
                - task_id: OpenML task ID
                - pipeline: Pipeline name
                - hp_id: Configuration ID
                - hp: Hyperparameter dict
                - status: Evaluation status
                - performance: Predictive accuracy
        """
        print(f"\nEvaluating {len(configurations)} configurations on task {task_id}...")
        
        # Convert configurations to SMAC Configuration objects
        init_configs = []
        for hp in configurations:
            try:
                cleaned_hp = self.clean_dict(hp)
                config = Configuration(
                    configuration_space=self.config_space, 
                    values=cleaned_hp
                )
                init_configs.append(config)
            except Exception as e:
                print(f"Warning: Could not create configuration: {e}")
                continue
        
        if len(init_configs) == 0:
            print("Error: No valid configurations to evaluate")
            return []
        
        # Create black-box function
        objective_function = self.generate_black_box_function(task_id=task_id, n_jobs=1)
        
        # Set up SMAC scenario
        scenario = Scenario({
            "run_obj": "quality",
            "runcount-limit": len(init_configs),
            "cs": self.config_space,
            "deterministic": "true",
            "execdir": "/tmp",
            "cutoff": 60 * 15,  # 15 minutes timeout
            "memory_limit": 5000,
            "cost_for_crash": 1,
            "abort_on_first_run_crash": False
        })
        
        # Create SMAC optimizer
        smac = SMAC4HPO(
            scenario=scenario,
            rng=np.random.RandomState(self.seed),
            tae_runner=objective_function,
            initial_configurations=init_configs,
            initial_design=None
        )
        
        # Register callback to collect results
        callback = ResultCallback(
            task_id=task_id, 
            pipeline=self.pipeline_name, 
            counter=counter
        )
        smac.register_callback(callback)
        
        # Run optimization
        smac.optimize()
        
        # Clean up SMAC output directory
        path = Path(smac.output_dir)
        parent = path.parent.absolute()
        shutil.rmtree(parent, ignore_errors=True)
        
        print(f"Evaluation complete: {len(callback.results)} results")
        
        return callback.results


def evaluate_task2_results(
    task_id,
    pipeline_name,
    config_space,
    sampled_configs,
    seed=42
):
    """
    Convenience function to evaluate Task 2 sampled configurations.
    
    Args:
        task_id: OpenML task ID
        pipeline_name: Name of the pipeline
        config_space: ConfigSpace object
        sampled_configs: DataFrame with sampled configurations
        seed: Random seed
    
    Returns:
        List of evaluation results
    """
    evaluator = ConfigurationEvaluator(pipeline_name, config_space, seed)
    
    # Convert DataFrame to list of dicts
    configs_list = sampled_configs.to_dict(orient='records')
    
    # Evaluate
    results = evaluator.evaluate_configurations(task_id, configs_list)
    
    return results


if __name__ == "__main__":
    print("This module provides the ConfigurationEvaluator class for evaluating hyperparameter configurations using SMAC.")