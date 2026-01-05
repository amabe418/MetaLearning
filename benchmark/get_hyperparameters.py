"""
Script para obtener hiperparámetros de OpenML dado un task, flow y AUC score.

Uso:
    python benchmark/get_hyperparameters.py --task-name anneal_task-1 --flow avNNet --auc 0.522177
"""

import argparse
import openml
import pandas as pd
from typing import Optional


def extract_task_id(task_name: str) -> int:
    """
    Extrae el task_id numérico del nombre de la tarea.
    Ej: 'anneal_task-1' -> 1
    """
    parts = task_name.split('-')
    if len(parts) >= 2:
        return int(parts[-1])
    raise ValueError(f"No se pudo extraer task_id de: {task_name}")


def find_run_by_auc(task_id: int, flow_name: str, target_auc: float, tolerance: float = 0.001) -> Optional[int]:
    """
    Busca el run_id que coincida con task_id, flow y AUC score.
    
    Args:
        task_id: ID de la tarea en OpenML
        flow_name: Nombre del flow (algoritmo)
        target_auc: Valor de area_under_roc_curve objetivo
        tolerance: Tolerancia para comparar AUC (default: 0.001)
    
    Returns:
        run_id si se encuentra, None si no
    """
    print(f"\nBuscando runs para task_id={task_id}, flow='{flow_name}', AUC≈{target_auc}")
    
    # Obtener todos los runs del task (límite: 20)
    runs = openml.runs.list_runs(task=[task_id], output_format='dataframe', size=20)
    
    if runs.empty:
        print(f"  ✗ No se encontraron runs para task_id={task_id}")
        return None
    
    print(f"  ✓ Encontrados {len(runs)} runs (límite: 20)")
    
    # Filtrar por flow_name si está disponible
    if 'flow_name' in runs.columns:
        runs = runs[runs['flow_name'].str.contains(flow_name, case=False, na=False)]
        print(f"  ✓ Filtrados a {len(runs)} runs con flow '{flow_name}'")
    
    # Buscar por AUC
    best_match = None
    min_diff = float('inf')
    
    for idx, run_id in enumerate(runs.index[:20], 1):  # Procesar máximo 20
        try:
            print(f"  Verificando run {idx}/20...", end='\r')
            run = openml.runs.get_run(run_id)
            evaluations = run.evaluations
            
            if 'area_under_roc_curve' in evaluations:
                auc = evaluations['area_under_roc_curve']
                diff = abs(auc - target_auc)
                
                if diff < min_diff:
                    min_diff = diff
                    best_match = run_id
                    
                    if diff <= tolerance:
                        print(f"  ✓ Match exacto encontrado: run_id={run_id}, AUC={auc:.6f}" + " "*20)
                        return run_id
        except Exception as e:
            continue
    
    print(" " * 50, end='\r')  # Limpiar línea
    
    if best_match:
        print(f"  ~ Mejor coincidencia: run_id={best_match}, diff={min_diff:.6f}")
        return best_match
    
    print(f"  ✗ No se encontró run con AUC cercano a {target_auc}")
    return None


def get_hyperparameters(run_id: int) -> dict:
    """
    Obtiene los hiperparámetros de un run específico.
    
    Args:
        run_id: ID del run en OpenML
    
    Returns:
        Diccionario con los hiperparámetros
    """
    print(f"\nObteniendo hiperparámetros del run_id={run_id}...")
    
    run = openml.runs.get_run(run_id)
    
    # Extraer hiperparámetros del setup
    if run.parameter_settings:
        hyperparams = {}
        for param in run.parameter_settings:
            if isinstance(param, dict):
                # Si es diccionario, buscar las claves correctas
                param_name = param.get('oml:name') or param.get('parameter_name') or param.get('name')
                param_value = param.get('oml:value') or param.get('value')
            else:
                # Si es objeto, usar atributos
                param_name = getattr(param, 'name', None) or getattr(param, 'parameter_name', None)
                param_value = getattr(param, 'value', None)
            
            if param_name and param_value is not None:
                hyperparams[param_name] = param_value
        
        print(f"  ✓ {len(hyperparams)} hiperparámetros encontrados")
        return hyperparams
    else:
        print(f"  ✗ No se encontraron hiperparámetros")
        return {}


def main():
    parser = argparse.ArgumentParser(
        description="Obtiene hiperparámetros de OpenML para una tarea, flow y AUC específicos"
    )
    
    parser.add_argument(
        '--task-name',
        type=str,
        required=True,
        help='Nombre de la tarea (ej: anneal_task-1)'
    )
    
    parser.add_argument(
        '--flow',
        type=str,
        required=True,
        help='Nombre del flow/algoritmo (ej: avNNet)'
    )
    
    parser.add_argument(
        '--auc',
        type=float,
        required=True,
        help='Valor de area_under_roc_curve (ej: 0.522177)'
    )
    
    parser.add_argument(
        '--tolerance',
        type=float,
        default=0.001,
        help='Tolerancia para comparar AUC (default: 0.001)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Archivo CSV para guardar los hiperparámetros (opcional)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("Búsqueda de Hiperparámetros en OpenML")
    print("="*70)
    
    # 1. Extraer task_id
    try:
        task_id = extract_task_id(args.task_name)
        print(f"\n[1/3] Task: {args.task_name} → task_id={task_id}")
    except ValueError as e:
        print(f"✗ Error: {e}")
        return
    
    # 2. Buscar run_id
    print(f"\n[2/3] Buscando run...")
    run_id = find_run_by_auc(task_id, args.flow, args.auc, args.tolerance)
    
    if not run_id:
        print("\n✗ No se pudo encontrar un run que coincida")
        return
    
    # 3. Obtener hiperparámetros
    print(f"\n[3/3] Extrayendo hiperparámetros...")
    hyperparams = get_hyperparameters(run_id)
    
    # Mostrar resultados
    print("\n" + "="*70)
    print("RESULTADOS")
    print("="*70)
    print(f"Task: {args.task_name} (task_id={task_id})")
    print(f"Flow: {args.flow}")
    print(f"AUC: {args.auc}")
    print(f"Run ID: {run_id}")
    print(f"\nHiperparámetros ({len(hyperparams)}):")
    print("-"*70)
    
    for param, value in sorted(hyperparams.items()):
        print(f"  {param:40s} = {value}")
    
    # Guardar si se especificó output
    if args.output and hyperparams:
        df = pd.DataFrame([{
            'task_name': args.task_name,
            'task_id': task_id,
            'flow': args.flow,
            'auc': args.auc,
            'run_id': run_id,
            **hyperparams
        }])
        df.to_csv(args.output, index=False)
        print(f"\n✓ Hiperparámetros guardados en: {args.output}")
    
    print("="*70)


if __name__ == "__main__":
    main()
