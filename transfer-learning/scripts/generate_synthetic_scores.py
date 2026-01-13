"""
Generador de Métricas de Rendimiento Sintéticas para FSBO
=========================================================

Este script genera métricas de rendimiento (scores) sintéticas pero realistas
para usar en el entrenamiento de FSBO cuando no se tienen datos reales.

Estrategia:
1. Crear una "superficie de respuesta" basada en los hiperparámetros
2. Cada tarea (task_id) tiene diferentes pesos -> diferentes óptimos
3. Agregar ruido gaussiano para simular variabilidad real
4. Generar scores en rango [0.5, 1.0] (típico de accuracy/AUC)

NOTA: Esto es válido para proyectos académicos de aprendizaje e implementación.
      Para publicaciones científicas se requieren datos reales.

Autor: Generado para proyecto académico de MetaLearning
"""

import pandas as pd
import numpy as np
from pathlib import Path
import hashlib


def generate_task_weights(task_id: int, n_features: int, seed: int = 42) -> np.ndarray:
    """
    Genera pesos únicos para cada tarea basados en su ID.
    Esto asegura que diferentes tareas tengan diferentes óptimos.
    """
    # Usar hash del task_id para generar seed único pero reproducible
    task_hash = int(hashlib.md5(f"{task_id}_{seed}".encode()).hexdigest()[:8], 16)
    rng = np.random.RandomState(task_hash)
    
    # Pesos entre -1 y 1, algunos más importantes que otros
    weights = rng.randn(n_features) * 0.5
    return weights


def generate_synthetic_score(
    X: np.ndarray, 
    task_id: int,
    base_score: float = 0.75,
    noise_std: float = 0.03,
    seed: int = 42
) -> np.ndarray:
    """
    Genera scores sintéticos realistas para una configuración de hiperparámetros.
    
    La función simula que:
    - Scores están en rango [0.5, 1.0] (típico de accuracy)
    - Diferentes combinaciones de HPs dan diferentes resultados
    - Hay ruido aleatorio (como en evaluaciones reales)
    - Cada tarea tiene diferentes óptimos
    
    Args:
        X: Matriz de hiperparámetros (ya normalizados)
        task_id: ID de la tarea
        base_score: Score base alrededor del cual varían los resultados
        noise_std: Desviación estándar del ruido
        seed: Semilla para reproducibilidad
    
    Returns:
        Array de scores sintéticos
    """
    n_samples, n_features = X.shape
    
    # Generar pesos únicos para esta tarea
    weights = generate_task_weights(task_id, n_features, seed)
    
    # Calcular score base: combinación lineal de features
    # Normalizar para que esté centrado en 0
    linear_component = X @ weights
    linear_component = (linear_component - linear_component.mean()) / (linear_component.std() + 1e-8)
    
    # Agregar componente no lineal (interacciones)
    if n_features > 1:
        # Interacción entre primeras features
        interaction = X[:, 0] * X[:, 1] if n_features > 1 else 0
        interaction = (interaction - interaction.mean()) / (interaction.std() + 1e-8)
    else:
        interaction = 0
    
    # Combinar componentes
    raw_score = 0.7 * linear_component + 0.3 * interaction
    
    # Escalar a rango deseado [0.5, 1.0]
    # Usar sigmoid para mapear a [0, 1] y luego escalar
    sigmoid_score = 1 / (1 + np.exp(-raw_score * 0.5))
    
    # Escalar a [0.5, 1.0] con base_score como centro
    scaled_score = 0.5 + sigmoid_score * 0.5
    
    # Ajustar para que el promedio esté cerca de base_score
    scaled_score = scaled_score - scaled_score.mean() + base_score
    
    # Agregar ruido gaussiano
    rng = np.random.RandomState(seed + task_id)
    noise = rng.randn(n_samples) * noise_std
    
    final_score = scaled_score + noise
    
    # Clipear a rango válido [0.5, 0.99]
    final_score = np.clip(final_score, 0.50, 0.99)
    
    return final_score


def process_representation_file(
    input_path: Path,
    output_path: Path,
    score_column: str = "accuracy",
    seed: int = 42
):
    """
    Procesa un archivo de representación y añade scores sintéticos.
    
    Args:
        input_path: Ruta al CSV de entrada
        output_path: Ruta al CSV de salida
        score_column: Nombre de la columna de score
        seed: Semilla para reproducibilidad
    """
    print(f"\n📂 Procesando: {input_path.name}")
    
    # Cargar datos
    df = pd.read_csv(input_path)
    print(f"   Filas: {len(df):,}")
    
    # Identificar columnas de features (todo excepto task_id)
    feature_cols = [col for col in df.columns if col != 'task_id']
    print(f"   Features: {len(feature_cols)}")
    
    # Obtener task_ids únicos
    unique_tasks = df['task_id'].unique()
    print(f"   Tareas únicas: {len(unique_tasks)}")
    
    # Generar scores por tarea
    scores = np.zeros(len(df))
    
    for task_id in unique_tasks:
        mask = df['task_id'] == task_id
        X_task = df.loc[mask, feature_cols].values
        
        # Generar scores para esta tarea
        task_scores = generate_synthetic_score(
            X_task, 
            task_id=int(task_id),
            base_score=0.75 + np.random.RandomState(seed + int(task_id)).uniform(-0.1, 0.1),
            noise_std=0.03,
            seed=seed
        )
        
        scores[mask] = task_scores
    
    # Añadir columna de scores
    df[score_column] = scores
    
    # Estadísticas
    print(f"   Score medio: {scores.mean():.4f}")
    print(f"   Score std: {scores.std():.4f}")
    print(f"   Score min: {scores.min():.4f}")
    print(f"   Score max: {scores.max():.4f}")
    
    # Guardar
    df.to_csv(output_path, index=False)
    print(f"   ✅ Guardado en: {output_path.name}")
    
    return df


def main():
    """Función principal que procesa todos los archivos de representación."""
    
    print("=" * 60)
    print("🔬 Generador de Métricas Sintéticas para FSBO")
    print("=" * 60)
    print("\n⚠️  NOTA: Estos datos son sintéticos para propósitos académicos.")
    print("    Para investigación publicable, use datos reales de OpenML.\n")
    
    # Rutas
    base_path = Path(__file__).parent.parent / "data"
    input_dir = base_path / "representation"
    output_dir = base_path / "representation_with_scores"
    
    # Crear directorio de salida
    output_dir.mkdir(exist_ok=True)
    
    # Archivos a procesar
    files = [
        "adaboost_target_representation.csv",
        "random_forest_target_representation.csv", 
        "libsvm_svc_target_representation.csv",
        "autosklearn_target_representation.csv"
    ]
    
    # Procesar cada archivo
    results = {}
    for filename in files:
        input_path = input_dir / filename
        if input_path.exists():
            output_filename = filename.replace(".csv", "_with_scores.csv")
            output_path = output_dir / output_filename
            
            df = process_representation_file(
                input_path=input_path,
                output_path=output_path,
                score_column="accuracy",
                seed=42
            )
            results[filename] = df
        else:
            print(f"⚠️  No encontrado: {filename}")
    
    # Resumen final
    print("\n" + "=" * 60)
    print("📊 RESUMEN")
    print("=" * 60)
    
    total_samples = 0
    total_tasks = set()
    
    for filename, df in results.items():
        n_samples = len(df)
        n_tasks = df['task_id'].nunique()
        total_samples += n_samples
        total_tasks.update(df['task_id'].unique())
        
        print(f"\n{filename}:")
        print(f"  - Muestras: {n_samples:,}")
        print(f"  - Tareas: {n_tasks}")
        print(f"  - Muestras/tarea (promedio): {n_samples/n_tasks:.1f}")
    
    print(f"\n🎯 TOTAL:")
    print(f"  - Muestras totales: {total_samples:,}")
    print(f"  - Tareas únicas totales: {len(total_tasks)}")
    
    print("\n✅ ¡Listo! Los archivos con scores están en:")
    print(f"   {output_dir}")
    
    # Verificación para FSBO
    print("\n" + "=" * 60)
    print("✔️  VERIFICACIÓN PARA FSBO")
    print("=" * 60)
    
    min_samples_per_task = 5
    valid_for_fsbo = True
    
    for filename, df in results.items():
        samples_per_task = df.groupby('task_id').size()
        min_samples = samples_per_task.min()
        
        if min_samples < min_samples_per_task:
            print(f"⚠️  {filename}: Algunas tareas tienen < {min_samples_per_task} muestras")
            valid_for_fsbo = False
        else:
            print(f"✅ {filename}: Mín {min_samples} muestras/tarea")
    
    if valid_for_fsbo:
        print("\n🎉 ¡Los datos cumplen los requisitos mínimos para FSBO!")
    else:
        print("\n⚠️  Algunos archivos podrían necesitar más datos por tarea.")


if __name__ == "__main__":
    main()

