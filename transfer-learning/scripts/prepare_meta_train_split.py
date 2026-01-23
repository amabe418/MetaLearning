#!/usr/bin/env python3
"""
Prepara los datos separados en meta-train y meta-test para evaluar el modelo completo.

Este script:
1. Lee los datasets de train/validation de data_model/
2. Filtra los datos de pipeline_representation/ para crear splits separados
3. Genera los archivos necesarios para entrenar FSBO solo con meta-train

Uso:
    python scripts/prepare_meta_train_split.py
    python scripts/prepare_meta_train_split.py --algorithm adaboost
    python scripts/prepare_meta_train_split.py --algorithm all

Autor: Proyecto académico MetaLearning
"""

import sys
import json
import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Algoritmos disponibles
AVAILABLE_ALGORITHMS = [
    'adaboost', 'bernoulli_nb', 'decision_tree', 'extra_trees',
    'gaussian_nb', 'hist_gradient_boosting', 'kneighbors', 'lda',
    'linear_svc', 'mlp', 'multinomial_nb', 'passive_aggressive',
    'qda', 'random_forest', 'sgd', 'svc'
]


def load_dataset_splits(base_dir: Path) -> tuple:
    """Carga las listas de datasets para train y validation."""
    train_file = base_dir / 'data_model' / 'train_datasets.txt'
    val_file = base_dir / 'data_model' / 'validation_datasets.txt'
    
    with open(train_file) as f:
        train_datasets = set(l.strip() for l in f if l.strip())
    
    with open(val_file) as f:
        val_datasets = set(l.strip() for l in f if l.strip())
    
    logger.info(f"Train datasets: {len(train_datasets)}")
    logger.info(f"Validation datasets: {len(val_datasets)}")
    
    return train_datasets, val_datasets


def prepare_algorithm_split(
    algorithm: str,
    train_datasets: set,
    val_datasets: set,
    data_dir: Path,
    output_dir: Path
):
    """
    Prepara los datos de un algoritmo separados en train/validation.
    
    Args:
        algorithm: Nombre del algoritmo (ej: 'adaboost')
        train_datasets: Set de nombres de datasets para train
        val_datasets: Set de nombres de datasets para validation
        data_dir: Directorio con pipeline_representation/
        output_dir: Directorio de salida
    """
    repr_file = data_dir / 'pipeline_representation' / f'{algorithm}_pipeline_representation.csv'
    mapping_file = data_dir / 'pipeline_representation' / f'{algorithm}_task_mapping.json'
    
    if not repr_file.exists():
        logger.warning(f"No encontrado: {repr_file}")
        return None
    
    # Cargar datos
    df = pd.read_csv(repr_file)
    
    with open(mapping_file) as f:
        task_mapping = json.load(f)
    
    # Crear mapeo inverso: task_id -> dataset_name
    inverse_mapping = {v: k for k, v in task_mapping.items()}
    
    # Identificar task_ids para train y validation
    train_task_ids = set()
    val_task_ids = set()
    
    for dataset_name, task_id in task_mapping.items():
        if dataset_name in train_datasets:
            train_task_ids.add(task_id)
        elif dataset_name in val_datasets:
            val_task_ids.add(task_id)
    
    # Filtrar datos
    df_train = df[df['task_id'].isin(train_task_ids)].copy()
    df_val = df[df['task_id'].isin(val_task_ids)].copy()
    
    logger.info(f"{algorithm}: train={len(df_train)} rows ({len(train_task_ids)} tasks), "
                f"val={len(df_val)} rows ({len(val_task_ids)} tasks)")
    
    # Crear directorios de salida
    train_output_dir = output_dir / 'meta_train'
    val_output_dir = output_dir / 'meta_test'
    train_output_dir.mkdir(parents=True, exist_ok=True)
    val_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar CSVs
    df_train.to_csv(train_output_dir / f'{algorithm}_pipeline_representation.csv', index=False)
    df_val.to_csv(val_output_dir / f'{algorithm}_pipeline_representation.csv', index=False)
    
    # Guardar mappings filtrados
    train_mapping = {k: v for k, v in task_mapping.items() if v in train_task_ids}
    val_mapping = {k: v for k, v in task_mapping.items() if v in val_task_ids}
    
    with open(train_output_dir / f'{algorithm}_task_mapping.json', 'w') as f:
        json.dump(train_mapping, f, indent=2)
    
    with open(val_output_dir / f'{algorithm}_task_mapping.json', 'w') as f:
        json.dump(val_mapping, f, indent=2)
    
    return {
        'algorithm': algorithm,
        'train_tasks': len(train_task_ids),
        'train_rows': len(df_train),
        'val_tasks': len(val_task_ids),
        'val_rows': len(df_val)
    }


def create_dataset_similarity_matrix(
    algorithm: str,
    data_dir: Path,
    output_dir: Path,
    train_datasets: set
):
    """
    Crea una matriz de similitud entre datasets basada en performance de configs.
    
    Esta es una alternativa a MetaFeatX cuando no tenemos metafeatures.
    Usa el "comportamiento" de las configuraciones en cada dataset.
    
    Args:
        algorithm: Nombre del algoritmo
        data_dir: Directorio con pipeline_representation/
        output_dir: Directorio de salida
        train_datasets: Solo usar datasets de train
    """
    repr_file = data_dir / 'pipeline_representation' / f'{algorithm}_pipeline_representation.csv'
    mapping_file = data_dir / 'pipeline_representation' / f'{algorithm}_task_mapping.json'
    
    if not repr_file.exists():
        return None
    
    df = pd.read_csv(repr_file)
    
    with open(mapping_file) as f:
        task_mapping = json.load(f)
    
    # Filtrar solo datasets de train
    train_task_ids = [v for k, v in task_mapping.items() if k in train_datasets]
    df_train = df[df['task_id'].isin(train_task_ids)]
    
    # Para cada dataset, crear un vector de "performance profile"
    # Esto es: para cada configuración vista, su accuracy
    feature_cols = [c for c in df.columns if c not in ['task_id', 'accuracy']]
    
    # Agrupar por task_id y calcular estadísticas
    task_profiles = df_train.groupby('task_id').agg({
        'accuracy': ['mean', 'std', 'min', 'max', 'count']
    }).reset_index()
    task_profiles.columns = ['task_id', 'acc_mean', 'acc_std', 'acc_min', 'acc_max', 'n_configs']
    task_profiles['acc_std'] = task_profiles['acc_std'].fillna(0)
    
    # Guardar perfiles de tareas (puede usarse como "metafeatures" alternativas)
    output_dir = output_dir / 'meta_train'
    output_dir.mkdir(parents=True, exist_ok=True)
    task_profiles.to_csv(output_dir / f'{algorithm}_task_profiles.csv', index=False)
    
    logger.info(f"{algorithm}: Creados perfiles para {len(task_profiles)} tareas")
    
    return task_profiles


def main():
    parser = argparse.ArgumentParser(
        description='Preparar datos para meta-train/meta-test split'
    )
    parser.add_argument('--algorithm', type=str, default='all',
                       choices=AVAILABLE_ALGORITHMS + ['all'])
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Directorio con pipeline_representation/')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directorio de salida')
    
    args = parser.parse_args()
    
    # Rutas
    base_dir = Path(__file__).parent.parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else base_dir / 'transfer-learning' / 'data'
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / 'splits'
    
    print("=" * 70)
    print("📊 PREPARACIÓN DE DATOS PARA META-TRAIN/META-TEST")
    print("=" * 70)
    print(f"\nConfiguración:")
    print(f"  Algoritmo: {args.algorithm}")
    print(f"  Data dir: {data_dir}")
    print(f"  Output dir: {output_dir}")
    
    # Cargar splits de datasets
    print("\n[1] Cargando splits de datasets...")
    train_datasets, val_datasets = load_dataset_splits(base_dir)
    
    # Algoritmos a procesar
    if args.algorithm == 'all':
        algorithms = AVAILABLE_ALGORITHMS
    else:
        algorithms = [args.algorithm]
    
    # Procesar cada algoritmo
    print("\n[2] Procesando algoritmos...")
    results = []
    
    for algorithm in algorithms:
        result = prepare_algorithm_split(
            algorithm=algorithm,
            train_datasets=train_datasets,
            val_datasets=val_datasets,
            data_dir=data_dir,
            output_dir=output_dir
        )
        
        if result:
            results.append(result)
            
            # También crear matriz de similitud
            create_dataset_similarity_matrix(
                algorithm=algorithm,
                data_dir=data_dir,
                output_dir=output_dir,
                train_datasets=train_datasets
            )
    
    # Resumen
    print("\n" + "=" * 70)
    print("📋 RESUMEN")
    print("=" * 70)
    
    if results:
        print(f"\n{'Algoritmo':<25} {'Train Tasks':<12} {'Train Rows':<12} {'Val Tasks':<12} {'Val Rows':<12}")
        print("-" * 70)
        for r in results:
            print(f"{r['algorithm']:<25} {r['train_tasks']:<12} {r['train_rows']:<12} "
                  f"{r['val_tasks']:<12} {r['val_rows']:<12}")
    
    print(f"\n📁 Datos guardados en:")
    print(f"   Meta-train: {output_dir / 'meta_train'}")
    print(f"   Meta-test:  {output_dir / 'meta_test'}")
    
    print("\n✅ ¡Preparación completada!")
    print("\nSiguientes pasos:")
    print("  1. Entrenar FSBO con datos de meta_train/")
    print("     python scripts/train_fsbo_pipeline.py --data_dir splits/meta_train --algorithm <algo>")
    print("  2. Evaluar en meta_test/")
    print("     python scripts/evaluate_meta_test.py --algorithm <algo>")


if __name__ == "__main__":
    main()
