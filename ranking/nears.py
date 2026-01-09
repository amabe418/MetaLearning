"""
Extractor de Vecinos usando el Modelo MetaFeatX.

Este módulo reutiliza la lógica de Task1 para extraer los vecinos más cercanos
de cualquier tarea usando las representaciones del modelo MetaFeatX.

El modelo MetaFeatX:
1. Entrena un mapeo lineal (ψ) aprendido que transforma metafeatures básicas
2. El espacio resultante preserva similitud de hiperparámetros óptimos
3. Las distancias euclidianas reflejan similitud de HP

"""

import os
import sys
import argparse
from pathlib import Path
from omegaconf import OmegaConf
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

# Agregar ruta del proyecto
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment.utils import (
    load_basic_features, 
    load_target_features, 
)

from experiment.reprs import get_model_representations

def get_neighbors_wrt_metabu_mf(test_task_id, pipeline, data_path, k=10, verbose=True):
    """
    Extrae los K vecinos más cercanos a una tarea usando el modelo MetaFeatX.
    
    Esta función reutiliza la lógica de Task1 para entrenar MetaFeatX y calcular
    vecinos. El modelo aprende un mapeo que preserva similitud de HP óptimos.
    
    Args:
        test_task_id (int): ID de la tarea para la cual encontrar vecinos
        pipeline (str): Nombre del pipeline (e.g., 'adaboost', 'random_forest')
        data_path (str): Ruta a los datos
        k (int): Número de vecinos a retornar (default=10)
        verbose (bool): Mostrar progreso (default=True)
    
    Returns:
        list: IDs de las K tareas más cercanas (ordenadas por distancia)
        
    Example:
        >>> neighbors = get_neighbors_wrt_metabu_mf(
        ...     test_task_id=3,
        ...     pipeline='adaboost',
        ...     data_path='./data_metabu_iclr',
        ...     k=10
        ... )
        >>> print(neighbors)
        [14952, 45, 9957, 49, 29, 146821, 31, 9977, 43, 37]
    """
    
    if verbose:
        print(f"\n[1] Cargando datos...")
    
    base_path = "./conf"

    cfg_main = OmegaConf.load(os.path.join(base_path, "config.yaml"))
    cfg_pipeline = OmegaConf.load(os.path.join(base_path, "pipeline", f"{pipeline}.yaml"))
    
    cfg_metafeature = OmegaConf.load(os.path.join(base_path, "metafeature", "metafeatx.yaml"))
   
    cfg_task = OmegaConf.load(os.path.join(base_path, "task", "task1.yaml"))

    cfg = OmegaConf.create({
        "seed": cfg_main.get("seed", 42),
        "pipeline": cfg_pipeline,
        "metafeature": cfg_metafeature,
        "task": cfg_task,
        "openml_tid": cfg_main.get("openml_tid", test_task_id),
        "data_path": cfg_main.get("data_path", "./data"),
        "output_file": cfg_main.get("output_file", None)
    })

    if verbose:
        print(f"[INFO] Configs cargadas para pipeline={pipeline}, metafeature=metafeatx, task_id={task_id}")


    # Cargar target representations
    target_reprs = load_target_features(pipeline=cfg_pipeline, path=data_path)
    list_ids = sorted(list(target_reprs["task_id"].unique()))
    
    if verbose:
        print(f"  ✓ Target representations: {len(list_ids)} tareas")
        print(f"  ✓ Test task: {test_task_id}")
    
    if test_task_id not in list_ids:
        raise ValueError(f"Task {test_task_id} not found in data")
    
    # Cargar metafeatures básicas
    basic_reprs = load_basic_features(metafeature=cfg.metafeature, path=data_path)
    basic_reprs = basic_reprs[basic_reprs.task_id.isin(list_ids)]
    
    if verbose:
        print(f"  ✓ Metafeatures básicas: {basic_reprs.shape}")
    
    # Definir train_ids: todos excepto test_task_id
    train_ids = [task_id for task_id in list_ids if task_id != test_task_id]
    test_ids = [test_task_id]
    
    if verbose:
        print(f"\n[2] Entrenando MetaFeatX...")
        print(f"  Train tasks: {len(train_ids)}")
        print(f"  Test tasks: 1")
    
    # Entrenar MetaFeatX (como en Task1)
    basic_reprs_metabu = get_model_representations(
        cfg=cfg,
        basic_reprs=basic_reprs,
        target_reprs=target_reprs,
        list_ids=list_ids,
        train_ids=train_ids,
        test_ids=test_ids
    )
    
    if verbose:
        print(f"  ✓ Representaciones MetaFeatX calculadas: {basic_reprs_metabu.shape}")
    
    # Establecer índice por task_id
    basic_reprs_metabu = basic_reprs_metabu.set_index("task_id")
    
    if verbose:
        print(f"\n[3] Calculando distancias...")
    
    # Calcular distancias euclidianas en espacio MetaFeatX
    # Solo entre test task y todas las demás
    test_repr = basic_reprs_metabu.loc[[test_task_id]].values
    all_reprs = basic_reprs_metabu.loc[list_ids].values
    
    distances = pairwise_distances(test_repr, all_reprs, metric='euclidean')[0]
    
    if verbose:
        print(f"  ✓ Matriz de distancias calculada")
    
    # Obtener índices ordenados por distancia
    # Excluir el primero (la tarea misma)
    sorted_indices = np.argsort(distances)
    
    # El primer índice es la tarea misma (distancia = 0)
    # Los siguientes son los vecinos
    neighbor_indices = sorted_indices[1:k+1]
    neighbor_task_ids = [list_ids[i] for i in neighbor_indices]
    neighbor_distances = distances[neighbor_indices]
    
    if verbose:
        print(f"\n[4] Resultados...")
        print(f"  ✓ {k} vecinos más cercanos encontrados:")
        for i, (task_id, dist) in enumerate(zip(neighbor_task_ids, neighbor_distances), 1):
            print(f"     {i:2d}. Task {task_id:6d} (distancia: {dist:.4f})")
    
    return neighbor_task_ids


def extract_neighbors_for_all_tasks(pipeline, data_path, k=10, output_dir='./outputs', verbose=True):
    """
    Extrae vecinos para TODAS las tareas disponibles en el dataset.
    
    Args:
        pipeline (str): Nombre del pipeline
        data_path (str): Ruta a los datos
        k (int): Número de vecinos por tarea
        output_dir (str): Directorio para guardar el CSV
        verbose (bool): Mostrar progreso
    
    Returns:
        pandas.DataFrame: DataFrame con task_id y sus vecinos
        
    Example:
        >>> df = extract_neighbors_for_all_tasks(
        ...     pipeline='adaboost',
        ...     data_path='./data_metabu_iclr',
        ...     k=10,
        ...     output_dir='./outputs'
        ... )
        >>> df.to_csv('neighbors_all.csv', index=False)
    """
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Extrayendo vecinos para TODAS las tareas")
        print(f"{'='*70}")
        print(f"Pipeline: {pipeline}")
        print(f"Número de vecinos: {k}")
    
    # Cargar lista de todas las tareas
    class PipelineConfig:
        def __init__(self, name):
            self.name = name
    
    pipeline_cfg = PipelineConfig(pipeline)
    target_reprs = load_target_features(pipeline=pipeline_cfg, path=data_path)
    list_ids = sorted(list(target_reprs["task_id"].unique()))
    
    if verbose:
        print(f"Total de tareas: {len(list_ids)}\n")
    
    neighbors_data = []
    
    # Para cada tarea, extraer vecinos
    for idx, test_task_id in enumerate(list_ids, 1):
        if verbose:
            print(f"[{idx}/{len(list_ids)}] Procesando task {test_task_id}...", end=' ', flush=True)
        
        try:
            neighbors = get_neighbors_wrt_metabu_mf(
                test_task_id=test_task_id,
                pipeline=pipeline,
                data_path=data_path,
                k=k,
                verbose=False
            )
            
            row = {"task_id": test_task_id}
            for j, neighbor_id in enumerate(neighbors, 1):
                row[f"neighbor_{j}"] = neighbor_id
            
            neighbors_data.append(row)
            
            if verbose:
                print("✓")
        
        except Exception as e:
            if verbose:
                print(f"✗ Error: {str(e)}")
            continue
    
    neighbors_df = pd.DataFrame(neighbors_data)
    
    # Guardar a CSV
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"neighbors_metabu_{pipeline}.csv")
        neighbors_df.to_csv(output_file, index=False)
        
        if verbose:
            print(f"\n✓ Guardado en: {output_file}")
    
    return neighbors_df


def main():
    """Función principal para usar desde línea de comandos."""
    
    parser = argparse.ArgumentParser(
        description="Extrae vecinos usando modelo MetaFeatX (reutiliza lógica de Task1)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Ruta a los datos"
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default="adaboost",
        help="Pipeline (default: adaboost)"
    )
    parser.add_argument(
        "--task",
        type=int,
        default=None,
        help="Task ID específica (si no se proporciona, procesa todas)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Número de vecinos (default: 10)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Directorio de salida"
    )
    
    args = parser.parse_args()
    
    # Validar
    if not os.path.exists(args.data_path):
        print(f"✗ Error: No existe {args.data_path}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("EXTRACTOR DE VECINOS - MODELO MetaFeatX")
    print("="*70)
    print(f"\nConfigración:")
    print(f"  Data: {args.data_path}")
    print(f"  Pipeline: {args.pipeline}")
    print(f"  K vecinos: {args.k}")
    print(f"  Output: {args.output_dir}")
    
    try:
        if args.task:
            # Una tarea específica
            print(f"  Tarea: {args.task}\n")
            neighbors = get_neighbors_wrt_metabu_mf(
                test_task_id=args.task,
                pipeline=args.pipeline,
                data_path=args.data_path,
                k=args.k,
                verbose=True
            )
            print(f"\nVecinos: {neighbors}")


            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                output_file = os.path.join(args.output_dir, f"neighbors_task_{args.task}_{args.pipeline}.csv")
                import pandas as pd
                row = {"task_id": args.task}
                for i, n in enumerate(neighbors, 1):
                    row[f"neighbor_{i}"] = n
                pd.DataFrame([row]).to_csv(output_file, index=False)
                print(f"\n✓ CSV guardado en: {output_file}")
                
        else:
            # Todas las tareas
            neighbors_df = extract_neighbors_for_all_tasks(
                pipeline=args.pipeline,
                data_path=args.data_path,
                k=args.k,
                output_dir=args.output_dir,
                verbose=True
            )
            
            print(f"\n{'='*70}")
            print("✓ COMPLETADO")
            print(f"{'='*70}")
            print(f"\nPrimeras 5 tareas:")
            print(neighbors_df.head())

        
    
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
