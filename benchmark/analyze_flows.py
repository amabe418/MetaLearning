"""
Script para analizar flows en OpenML y seleccionar los 30 más frecuentes.

Uso:
    python benchmark/analyze_flows.py --benchmark openml_cc18
    python benchmark/analyze_flows.py --benchmark openml_100 --top-n 30
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter
from typing import List, Dict

import openml
from tqdm import tqdm

# Agregar config al path
sys.path.insert(0, str(Path(__file__).parent.parent / 'config'))
from config import benchmarks


def get_flows_for_dataset(dataset_id: int) -> List[int]:
    """
    Obtiene todos los flow IDs que se han ejecutado en un dataset.
    
    Args:
        dataset_id: ID del dataset en OpenML
        
    Returns:
        Lista de flow IDs ejecutados en ese dataset
    """
    try:
        # Obtener task para el dataset
        task = openml.tasks.get_task(dataset_id)
        
        # Obtener todas las evaluaciones para este task
        evaluations = openml.evaluations.list_evaluations(
            function='predictive_accuracy',
            tasks=[task.task_id],
            output_format='dataframe'
        )
        
        if evaluations is None or evaluations.empty:
            print(f"  ⚠️  Dataset {dataset_id}: No evaluations found")
            return []
        
        # Extraer flow IDs únicos
        flow_ids = evaluations['flow_id'].unique().tolist()
        print(f"  ✓ Dataset {dataset_id}: {len(flow_ids)} flows found")
        return flow_ids
        
    except Exception as e:
        print(f"  ✗ Dataset {dataset_id}: Error - {e}")
        return []


def get_flow_name(flow_id: int) -> str:
    """
    Obtiene el nombre legible de un flow.
    
    Args:
        flow_id: ID del flow en OpenML
        
    Returns:
        Nombre del flow
    """
    try:
        flow = openml.flows.get_flow(flow_id)
        return flow.name
    except Exception as e:
        print(f"  Warning: Could not get name for flow {flow_id}: {e}")
        return f"flow_{flow_id}"


def analyze_flows(dataset_ids: List[int], top_n: int = 30) -> Dict[int, str]:
    """
    Analiza flows en múltiples datasets y selecciona los más frecuentes.
    
    Args:
        dataset_ids: Lista de dataset IDs a analizar
        top_n: Número de flows a seleccionar
        
    Returns:
        Diccionario {flow_id: flow_name} con los top N flows
    """
    print(f"\n{'='*60}")
    print(f"Analizando flows en {len(dataset_ids)} datasets...")
    print(f"{'='*60}\n")
    
    all_flows = []
    
    # Recopilar flows de todos los datasets
    for dataset_id in tqdm(dataset_ids, desc="Procesando datasets"):
        flows = get_flows_for_dataset(dataset_id)
        all_flows.extend(flows)
    
    # Contar frecuencias
    print(f"\n{'='*60}")
    print("Contando frecuencias de flows...")
    print(f"{'='*60}\n")
    
    flow_counter = Counter(all_flows)
    total_unique_flows = len(flow_counter)
    
    print(f"Total de flows únicos encontrados: {total_unique_flows}")
    print(f"Total de ejecuciones: {len(all_flows)}\n")
    
    # Seleccionar top N
    top_flows = flow_counter.most_common(top_n)
    
    print(f"{'='*60}")
    print(f"Top {top_n} Flows más frecuentes:")
    print(f"{'='*60}\n")
    
    # Obtener nombres de los flows
    top_flows_dict = {}
    for rank, (flow_id, count) in enumerate(top_flows, 1):
        flow_name = get_flow_name(flow_id)
        top_flows_dict[flow_id] = flow_name
        
        coverage = (count / len(dataset_ids)) * 100
        print(f"{rank:2d}. Flow ID {flow_id:4d} - {count:3d} datasets ({coverage:5.1f}%) - {flow_name}")
    
    return top_flows_dict


def get_datasets_from_benchmark(benchmark_name: str) -> List[int]:
    """
    Obtiene la lista de datasets desde la configuración de benchmarks.
    
    Args:
        benchmark_name: Nombre del benchmark en benchmarks.yaml
        
    Returns:
        Lista de dataset IDs
    """
    if benchmark_name not in benchmarks:
        print(f"Error: Benchmark '{benchmark_name}' no encontrado")
        print(f"Benchmarks disponibles: {list(benchmarks.keys())}")
        sys.exit(1)
    
    bench_cfg = benchmarks[benchmark_name]
    
    try:
        # Obtener suite de OpenML
        if "suite_id" in bench_cfg:
            suite = openml.study.get_suite(bench_cfg["suite_id"])
        elif "suite_name" in bench_cfg:
            suite = openml.study.get_suite(bench_cfg["suite_name"])
        else:
            raise ValueError("Benchmark config must have 'suite_id' or 'suite_name'")
        
        # Obtener dataset IDs desde tasks
        dataset_ids = []
        print(f"Obteniendo datasets de {len(suite.tasks)} tasks...")
        for task_id in tqdm(suite.tasks, desc="Procesando tasks"):
            try:
                task = openml.tasks.get_task(task_id)
                dataset_ids.append(task.dataset_id)
            except Exception as e:
                print(f"  Warning: Error con task {task_id}: {e}")
        
        return dataset_ids
        
    except Exception as e:
        print(f"Error obteniendo datasets del benchmark: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Analiza flows en OpenML y selecciona los más frecuentes"
    )
    
    parser.add_argument(
        '--benchmark',
        type=str,
        required=True,
        help='Nombre del benchmark desde config/benchmarks.yaml (ej: openml_cc18, openml_100)'
    )
    
    parser.add_argument(
        '--top-n',
        type=int,
        default=30,
        help='Número de flows a seleccionar (default: 30)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/top_30_flows.json',
        help='Ruta del archivo de salida JSON'
    )
    
    args = parser.parse_args()
    
    # Obtener datasets del benchmark
    print(f"Usando benchmark: {args.benchmark}")
    dataset_ids = get_datasets_from_benchmark(args.benchmark)
    
    print(f"\nDatasets a analizar: {len(dataset_ids)}")
    print(f"Primeros 10: {dataset_ids[:10]}...\n")
    
    # Analizar flows
    top_flows = analyze_flows(dataset_ids, top_n=args.top_n)
    
    # Guardar resultados
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(top_flows, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✓ Resultados guardados en: {output_path}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
