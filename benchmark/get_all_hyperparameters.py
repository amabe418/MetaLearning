"""
Script para obtener hiperparámetros de todas las filas de original_datasets.csv

Uso:
    python benchmark/get_all_hyperparameters.py --input data/original_datasets.csv --output data/hyperparameters.csv
"""

import argparse
import time
import pandas as pd
from pathlib import Path
from get_hyperparameters import extract_task_id, find_run_by_auc, get_hyperparameters


def process_all_rows(input_csv: Path, output_csv: Path, tolerance: float = 0.001, delay: float = 1.0):
    """
    Procesa todas las filas del CSV para obtener hiperparámetros.
    
    Args:
        input_csv: Ruta al archivo original_datasets.csv
        output_csv: Ruta al archivo de salida con hiperparámetros
        tolerance: Tolerancia para comparar AUC
        delay: Segundos de espera entre requests a OpenML (evitar rate limiting)
    """
    # Leer archivo de entrada
    df = pd.read_csv(input_csv)
    
    print(f"Total de filas a procesar: {len(df)}")
    print(f"Columnas: {list(df.columns)}")
    print("="*70)
    
    results = []
    
    for idx, row in df.iterrows():
        task_name = row['openml_task']
        flow = row['flow']
        auc = row['area_under_roc_curve']
        
        print(f"\n[{idx+1}/{len(df)}] Procesando: {task_name} | {flow} | AUC={auc:.6f}")
        
        try:
            # Extraer task_id
            task_id = extract_task_id(task_name)
            
            # Buscar run_id
            run_id = find_run_by_auc(task_id, flow, auc, tolerance)
            
            if run_id:
                # Obtener hiperparámetros
                hyperparams = get_hyperparameters(run_id)
                
                # Agregar a resultados
                result_row = {
                    'openml_task': task_name,
                    'task_id': task_id,
                    'flow': flow,
                    'area_under_roc_curve': auc,
                    'run_id': run_id,
                    **hyperparams,
                    **{col: row[col] for col in df.columns if col not in ['openml_task', 'flow', 'area_under_roc_curve']}
                }
                results.append(result_row)
                
                print(f"  ✓ Procesado exitosamente ({len(hyperparams)} hiperparámetros)")
            else:
                print(f"  ✗ No se encontró run_id")
                # Agregar fila sin hiperparámetros
                result_row = {
                    'openml_task': task_name,
                    'task_id': task_id,
                    'flow': flow,
                    'area_under_roc_curve': auc,
                    'run_id': None,
                    **{col: row[col] for col in df.columns if col not in ['openml_task', 'flow', 'area_under_roc_curve']}
                }
                results.append(result_row)
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
            # Agregar fila con error
            result_row = {
                'openml_task': task_name,
                'task_id': None,
                'flow': flow,
                'area_under_roc_curve': auc,
                'run_id': None,
                'error': str(e),
                **{col: row[col] for col in df.columns if col not in ['openml_task', 'flow', 'area_under_roc_curve']}
            }
            results.append(result_row)
        
        # Guardar progreso cada 10 filas
        if (idx + 1) % 10 == 0 and results:
            temp_df = pd.DataFrame(results)
            temp_df.to_csv(output_csv, index=False)
            print(f"\n  💾 Progreso guardado: {len(results)} filas procesadas")
        
        # Delay para evitar rate limiting
        time.sleep(delay)
    
    # Guardar resultados finales
    if results:
        final_df = pd.DataFrame(results)
        final_df.to_csv(output_csv, index=False)
        print("\n" + "="*70)
        print(f"✓ Proceso completado: {len(results)} filas guardadas en {output_csv}")
        print(f"  - Con hiperparámetros: {final_df['run_id'].notna().sum()}")
        print(f"  - Sin hiperparámetros: {final_df['run_id'].isna().sum()}")
        print("="*70)
    else:
        print("\n✗ No se procesaron filas")


def main():
    parser = argparse.ArgumentParser(
        description="Obtiene hiperparámetros para todas las filas de original_datasets.csv"
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Archivo CSV de entrada (original_datasets.csv)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Archivo CSV de salida con hiperparámetros'
    )
    
    parser.add_argument(
        '--tolerance',
        type=float,
        default=0.001,
        help='Tolerancia para comparar AUC (default: 0.001)'
    )
    
    parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help='Segundos entre requests a OpenML (default: 1.0)'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"✗ Error: No se encuentra el archivo {input_path}")
        return
    
    print("="*70)
    print("Obtención Masiva de Hiperparámetros desde OpenML")
    print("="*70)
    
    process_all_rows(input_path, output_path, args.tolerance, args.delay)


if __name__ == "__main__":
    main()
