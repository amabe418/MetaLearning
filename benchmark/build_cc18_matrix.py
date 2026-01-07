"""
Script para construir la matriz de performance del benchmark OpenML-CC18.

Ejecuta todo el proceso:
1. Obtiene datasets de OpenML-CC18
2. Analiza flows y selecciona los top 30 más usados
3. Construye tabla de performance (dataset, flow, accuracy)

Uso:
    python benchmark/build_cc18_matrix.py
"""

import json
import sys
from pathlib import Path
from collections import Counter
from typing import List, Dict, Optional

import openml
import pandas as pd
from tqdm import tqdm

# Configuración fija
TOP_N_FLOWS = 30
DEFAULT_ACCURACY = 0.0
OUTPUT_CSV = "data/processed/cc18_performance.csv"
OUTPUT_FLOWS = "data/processed/cc18_flows.json"

# Agregar directorio raíz al path para importar config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import benchmarks


def get_cc18_datasets() -> List[int]:
    """
    Obtiene la lista de datasets del benchmark OpenML-CC18.
    
    Returns:
        Lista de dataset IDs
    """
    print(f"\n{'='*60}")
    print("Obteniendo datasets de OpenML-CC18...")
    print(f"{'='*60}\n")
    
    bench_cfg = benchmarks["openml_cc18"]
    
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
        print(f"Procesando {len(suite.tasks)} tasks...")
        for task_id in tqdm(suite.tasks, desc="Obteniendo datasets"):
            try:
                task = openml.tasks.get_task(task_id)
                dataset_ids.append(task.dataset_id)
            except Exception as e:
                print(f"  Warning: Error con task {task_id}: {e}")
        
        print(f"\n✓ {len(dataset_ids)} datasets obtenidos")
        return dataset_ids
        
    except Exception as e:
        print(f"Error obteniendo datasets: {e}")
        sys.exit(1)


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
            return []
        
        # Extraer flow IDs únicos
        flow_ids = evaluations['flow_id'].unique().tolist()
        return flow_ids
        
    except Exception as e:
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
    for dataset_id in tqdm(dataset_ids, desc="Recopilando flows"):
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


def get_best_accuracy_for_flow(dataset_id: int, flow_id: int) -> Optional[float]:
    """
    Obtiene el mejor accuracy para un flow en un dataset específico.
    
    Args:
        dataset_id: ID del dataset en OpenML
        flow_id: ID del flow en OpenML
        
    Returns:
        Mejor accuracy encontrado, o None si no hay runs
    """
    try:
        # Obtener el task_id del dataset
        task = openml.tasks.get_task(dataset_id)
        task_id = task.task_id
        
        # Obtener evaluaciones para este task y flow
        evaluations = openml.evaluations.list_evaluations(
            function='predictive_accuracy',
            tasks=[task_id],
            flows=[flow_id],
            output_format='dataframe'
        )
        
        if evaluations is None or evaluations.empty:
            return None
        
        # Obtener el mejor accuracy
        best_accuracy = evaluations['value'].max()
        return float(best_accuracy)
        
    except Exception as e:
        return None


def build_performance_table(
    dataset_ids: List[int],
    flows: Dict[int, str],
    default_accuracy: float = 0.0
) -> pd.DataFrame:
    """
    Construye la tabla de performance para datasets y flows.
    
    Args:
        dataset_ids: Lista de dataset IDs
        flows: Diccionario {flow_id: flow_name}
        default_accuracy: Accuracy a usar cuando no hay runs (default: 0.0)
        
    Returns:
        DataFrame con columnas [dataset_id, flow_name, accuracy]
    """
    print(f"\n{'='*60}")
    print(f"Construyendo tabla de performance...")
    print(f"  Datasets: {len(dataset_ids)}")
    print(f"  Flows: {len(flows)}")
    print(f"  Combinaciones totales: {len(dataset_ids) * len(flows)}")
    print(f"{'='*60}\n")
    
    results = []
    total_combinations = len(dataset_ids) * len(flows)
    found_count = 0
    missing_count = 0
    
    # Crear barra de progreso para todas las combinaciones
    with tqdm(total=total_combinations, desc="Construyendo tabla") as pbar:
        for dataset_id in dataset_ids:
            for flow_id, flow_name in flows.items():
                # Obtener accuracy
                accuracy = get_best_accuracy_for_flow(dataset_id, flow_id)
                
                if accuracy is None:
                    accuracy = default_accuracy
                    missing_count += 1
                else:
                    found_count += 1
                
                # Agregar resultado
                results.append({
                    'dataset_id': dataset_id,
                    'flow_name': flow_name,
                    'accuracy': accuracy
                })
                
                pbar.update(1)
    
    # Crear DataFrame
    df = pd.DataFrame(results)
    
    print(f"\n{'='*60}")
    print("Estadísticas:")
    print(f"{'='*60}")
    print(f"  Combinaciones procesadas: {total_combinations}")
    print(f"  Runs encontrados: {found_count} ({found_count/total_combinations*100:.1f}%)")
    print(f"  Runs faltantes (penalizados): {missing_count} ({missing_count/total_combinations*100:.1f}%)")
    print(f"  Accuracy promedio: {df['accuracy'].mean():.4f}")
    print(f"  Accuracy mediana: {df['accuracy'].median():.4f}")
    print(f"{'='*60}\n")
    
    return df


def main():
    print(f"\n{'='*60}")
    print("CONSTRUCCIÓN DE MATRIZ DE PERFORMANCE - OpenML-CC18")
    print(f"{'='*60}")
    print(f"Top flows a seleccionar: {TOP_N_FLOWS}")
    print(f"Penalización para flows faltantes: {DEFAULT_ACCURACY}")
    print(f"{'='*60}\n")
    
    # Paso 1: Obtener datasets de CC18
    dataset_ids = get_cc18_datasets()
    
    # Paso 2: Analizar y seleccionar top 30 flows
    top_flows = analyze_flows(dataset_ids, top_n=TOP_N_FLOWS)
    
    # Guardar flows seleccionados
    flows_path = Path(OUTPUT_FLOWS)
    flows_path.parent.mkdir(parents=True, exist_ok=True)
    with open(flows_path, 'w') as f:
        json.dump(top_flows, f, indent=2)
    print(f"\n✓ Flows seleccionados guardados en: {flows_path}")
    
    # Paso 3: Construir tabla de performance
    df = build_performance_table(
        dataset_ids=dataset_ids,
        flows=top_flows,
        default_accuracy=DEFAULT_ACCURACY
    )
    
    # Guardar tabla de performance
    output_path = Path(OUTPUT_CSV)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Tabla de performance guardada en: {output_path}")
    print(f"  Forma: {df.shape[0]} filas × {df.shape[1]} columnas")
    
    # Mostrar muestra
    print(f"\nPrimeras 15 filas:")
    print(df.head(15).to_string(index=False))
    
    print(f"\n{'='*60}")
    print("¡PROCESO COMPLETADO!")
    print(f"  Archivo principal: {output_path}")
    print(f"  Flows seleccionados: {flows_path}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
