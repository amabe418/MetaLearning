"""
Módulo para cargar y gestionar datasets de OpenML.
"""

import openml
from openml.tasks import TaskType
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import os


def load_openml_dataset(dataset_id: int, cache_dir: Optional[str] = None) -> Dict:
    """
    Carga un dataset de OpenML por su ID.
    
    Args:
        dataset_id: ID del dataset en OpenML
        cache_dir: Directorio para cachear datasets (opcional)
    
    Returns:
        Diccionario con el dataset y metadatos
    """
    if cache_dir:
        openml.config.set_cache_directory(cache_dir)
    
    try:
        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target=dataset.default_target_attribute
        )
        
        return {
            'id': dataset_id,
            'name': dataset.name,
            'X': X,
            'y': y,
            'categorical_indicator': categorical_indicator,
            'attribute_names': attribute_names,
            'metadata': {
                'n_samples': len(X),
                'n_features': len(attribute_names),
                'n_classes': len(np.unique(y)) if y is not None else None,
                'task_type': dataset.qualities.get('NumberOfClasses', 'unknown'),
                'default_target': dataset.default_target_attribute
            }
        }
    except Exception as e:
        print(f"Error cargando dataset {dataset_id}: {e}")
        return None


def load_openml_datasets(dataset_ids: List[int], cache_dir: Optional[str] = None) -> List[Dict]:
    """
    Carga múltiples datasets de OpenML.
    
    Args:
        dataset_ids: Lista de IDs de datasets
        cache_dir: Directorio para cachear datasets (opcional)
    
    Returns:
        Lista de diccionarios con datasets
    """
    datasets = []
    for dataset_id in dataset_ids:
        dataset = load_openml_dataset(dataset_id, cache_dir)
        if dataset:
            datasets.append(dataset)
    return datasets

TASK_TYPES = {
    'Supervised Classification': TaskType.SUPERVISED_CLASSIFICATION,
    'Supervised Regression': TaskType.SUPERVISED_REGRESSION,
    'Clustering': TaskType.CLUSTERING,
}

def search_openml_datasets(
    task_type: str = 'Supervised Classification',
    n_samples_min: int = 100,
    n_samples_max: int = 10000,
    n_features_max: int = 100
) -> pd.DataFrame:
    """
    Busca datasets en OpenML según criterios.
    
    Args:
        task_type: Tipo de tarea ('Supervised Classification', 'Supervised Regression')
        n_samples_min: Número mínimo de muestras
        n_samples_max: Número máximo de muestras
        n_features_max: Número máximo de características
    
    Returns:
        DataFrame con información de datasets encontrados
    """

    if task_type not in TASK_TYPES:
        raise ValueError(f"Tipo de tarea desconocido: {task_type}") 
    
    print(TASK_TYPES[task_type])

    tasks = openml.tasks.list_tasks(
        task_type=TASK_TYPES[task_type],
        output_format='dataframe'
    )
    
    # Filtrar por criterios
    filtered = tasks[
        (tasks['NumberOfInstances'] >= n_samples_min) &
        (tasks['NumberOfInstances'] <= n_samples_max) &
        (tasks['NumberOfFeatures'] <= n_features_max)
    ]
    
    return filtered


def get_dataset_metadata(dataset_id: int) -> Dict:
    """
    Obtiene solo los metadatos de un dataset sin cargar los datos completos.
    
    Args:
        dataset_id: ID del dataset en OpenML
    
    Returns:
        Diccionario con metadatos del dataset
    """
    try:
        dataset = openml.datasets.get_dataset(dataset_id, download_data=False)
        return {
            'id': dataset_id,
            'name': dataset.name,
            'version': dataset.version,
            'description': dataset.description,
            'format': dataset.format,
            'qualities': dataset.qualities,
            'features': [f.name for f in dataset.features]
        }
    except Exception as e:
        print(f"Error obteniendo metadatos del dataset {dataset_id}: {e}")
        return None

