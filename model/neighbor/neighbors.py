#!/usr/bin/env python3
"""
CLI para extracción de vecinos MetaFeatX.

Ejemplo de uso:

1. Task específica:
    python -m model.neighbor.neighbors --data-path ./data --pipeline "" --task 3 --k 10 --output-dir ./outputs

"""

import os
import sys
import argparse
from pathlib import Path

# Agregar raíz del proyecto para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.neighbor.score import calculate_general_score
from model.neighbor.nears import get_neighbors_wrt_metabu_mf, extract_neighbors_for_all_tasks

def main():
    parser = argparse.ArgumentParser(description="Extractor de vecinos MetaFeatX CLI-like")
    parser.add_argument("--data-path", type=str, required=True, help="Ruta a los datos")
    parser.add_argument("--pipeline", type=str, default="", help="Pipeline específico, o vacío para todos")
    parser.add_argument("--task", type=int, default=None, help="Task específica, si no se da procesa todas")
    parser.add_argument("--k", type=int, default=10, help="Número de vecinos")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Directorio de salida")
    parser.add_argument("--verbose", action="store_true", help="Mostrar progreso")
    
    args = parser.parse_args()

    # Lista de pipelines
    all_pipelines = ['adaboost', 'random_forest', 'libsvm_svc']
    pipelines_to_run = [args.pipeline] if args.pipeline else all_pipelines
    all_neighbors = {}  # { "adaboost": {task_id: [vecinos]}, "random_forest": {...} }

    for pl in pipelines_to_run:
        if args.verbose:
            print("\n" + "="*70)
            print(f"Pipeline: {pl}")
            print("="*70)

        pipeline_neighbors = {}

        if args.task is not None:
            # Task específica
            neighbors = get_neighbors_wrt_metabu_mf(
                test_task_id=args.task,
                pipeline=pl,
                data_path=args.data_path,
                k=args.k,
                verbose=args.verbose
            )
            print(f"\nVecinos de task_id={args.task}: {neighbors}")

            pipeline_neighbors[args.task] = neighbors

            # Guardar CSV
            # if args.output_dir:
            #     os.makedirs(args.output_dir, exist_ok=True)
            #     output_file = os.path.join(args.output_dir, f"neighbors_task_{args.task}_{pl}.csv")
            #     import pandas as pd
            #     row = {"task_id": args.task}
            #     for i, n in enumerate(neighbors, 1):
            #         row[f"neighbor_{i}"] = n
            #     pd.DataFrame([row]).to_csv(output_file, index=False)
            #     if args.verbose:
            #         print(f"✓ CSV guardado en: {output_file}")



            

        else:
            # Todas las tasks
            neighbors_df = extract_neighbors_for_all_tasks(
                pipeline=pl,
                data_path=args.data_path,
                k=args.k,
                output_dir=args.output_dir,
                verbose=args.verbose
            )
            if args.verbose:
                print(f"\nPrimeras 5 filas:\n{neighbors_df.head()}")

        all_neighbors[pl] = pipeline_neighbors

    print(f"Para {args.task} ranking es:\n")

    print(calculate_general_score(all_neighbors,task_id=args.task, k = 20))       

if __name__ == "__main__":
    main()
