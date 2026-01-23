"""
Script de prueba para validar el flujo completo de FSBO con pipeline.

Este script ejecuta todo el flujo:
1. Prepara los datos
2. Entrena el modelo (pocas épocas para prueba)
3. Usa el optimizador para sugerir configuraciones

Uso:
    python scripts/test_pipeline_flow.py --algorithm adaboost

Autor: Proyecto académico MetaLearning
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def test_data_preparation(algorithm: str) -> bool:
    """Prueba la preparación de datos."""
    print("\n" + "=" * 50)
    print("📦 PASO 1: Preparación de Datos")
    print("=" * 50)
    
    base_dir = Path(__file__).parent.parent.parent
    input_dir = base_dir / 'pipes' / 'combined'
    output_dir = base_dir / 'transfer-learning' / 'data' / 'pipeline_representation'
    
    # Verificar datos de entrada
    csv_mapping = {
        'adaboost': 'AdaBoostClassifier',
        'random_forest': 'RandomForestClassifier',
        'svc': 'SVC',
    }
    
    csv_name = csv_mapping.get(algorithm, algorithm.replace('_', ''))
    input_file = input_dir / f"{csv_name}_combined.csv"
    
    if not input_file.exists():
        print(f"❌ No encontrado: {input_file}")
        return False
    
    print(f"✅ Datos de entrada encontrados: {input_file.name}")
    
    # Ejecutar preparación
    from prepare_pipeline_data import prepare_data_for_algorithm, generate_configspace, ALGORITHM_MAPPING
    
    # Encontrar nombre interno
    internal_name = None
    for csv_n, int_n in ALGORITHM_MAPPING.items():
        if int_n == algorithm:
            internal_name = int_n
            break
    
    if internal_name is None:
        internal_name = algorithm
    
    output_dir.mkdir(parents=True, exist_ok=True)
    configspace_dir = base_dir / 'transfer-learning' / 'data' / 'pipeline_configspace'
    configspace_dir.mkdir(parents=True, exist_ok=True)
    
    result = prepare_data_for_algorithm(input_file, internal_name, output_dir)
    
    if result:
        print(f"✅ Datos preparados: {result['n_samples']} muestras, {result['n_features']} features")
        generate_configspace(internal_name, result['feature_names'], configspace_dir)
        print(f"✅ ConfigSpace generado")
        return True
    else:
        print("❌ Error preparando datos")
        return False


def test_training(algorithm: str, epochs: int = 100) -> bool:
    """Prueba el entrenamiento (pocas épocas)."""
    print("\n" + "=" * 50)
    print("🎓 PASO 2: Entrenamiento (prueba rápida)")
    print("=" * 50)
    
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data' / 'pipeline_representation'
    checkpoint_dir = base_dir / 'experiments' / 'checkpoints_pipeline'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    data_file = data_dir / f"{algorithm}_pipeline_representation.csv"
    
    if not data_file.exists():
        print(f"❌ Datos no encontrados: {data_file}")
        return False
    
    print(f"✅ Datos encontrados: {data_file.name}")
    print(f"   Entrenando con {epochs} épocas (prueba rápida)...")
    
    from train_fsbo_pipeline import (
        PipelineMetaDataset, train_fsbo_pipeline
    )
    import torch
    from datetime import datetime
    
    # Cargar datos
    dataset = PipelineMetaDataset(min_evaluations=3)
    dataset.load_from_csv(str(data_file))
    dataset.split_tasks()
    
    print(f"   Tareas: {len(dataset.tasks)}, Input dim: {dataset.get_input_dim()}")
    
    # Entrenar
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, likelihood, losses = train_fsbo_pipeline(
        dataset=dataset,
        n_iterations=epochs,
        batch_size=32,
        lr=1e-3,
        hidden_dim=64,  # Más pequeño para prueba
        n_layers=2,
        device=device
    )
    
    # Guardar checkpoint
    checkpoint_path = checkpoint_dir / f"fsbo_pipeline_{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    torch.save({
        'model_state': model.state_dict(),
        'likelihood_state': likelihood.state_dict(),
        'losses': losses,
        'config': {
            'algorithm': algorithm,
            'input_dim': dataset.get_input_dim(),
            'hidden_dim': 64,
            'n_layers': 2,
            'feature_names': dataset.feature_names,
        }
    }, checkpoint_path)
    
    final_loss = np.mean(losses[-20:]) if len(losses) >= 20 else np.mean(losses)
    print(f"✅ Entrenamiento completado!")
    print(f"   Loss final: {final_loss:.4f}")
    print(f"   Checkpoint: {checkpoint_path.name}")
    
    return True


def test_optimization(algorithm: str) -> bool:
    """Prueba la optimización."""
    print("\n" + "=" * 50)
    print("🎯 PASO 3: Optimización (prueba)")
    print("=" * 50)
    
    from fsbo_pipeline_optimizer import FSBOPipelineOptimizer
    
    try:
        optimizer = FSBOPipelineOptimizer.from_pretrained(algorithm)
        print(f"✅ Optimizador cargado para {algorithm}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return False
    
    # Simular optimización
    print("   Sugiriendo configuraciones iniciales...")
    initial_configs = optimizer.suggest_initial(n=3)
    
    print(f"   Configuraciones iniciales sugeridas: {len(initial_configs)}")
    for i, config in enumerate(initial_configs):
        print(f"\n   Config {i+1}:")
        print(f"     Pipeline: {config['pipeline']}")
        print(f"     Classifier: {config['classifier']}")
        
        # Simular evaluación
        dummy_score = 0.7 + np.random.uniform(-0.1, 0.2)
        optimizer.observe(config, dummy_score)
        print(f"     Score (simulado): {dummy_score:.4f}")
    
    # Sugerir siguiente
    print("\n   Sugiriendo siguiente configuración...")
    next_config = optimizer.suggest()
    print(f"   Siguiente config sugerida:")
    print(f"     Pipeline: {next_config['pipeline']}")
    print(f"     Classifier: {next_config['classifier']}")
    
    # Mejor hasta ahora
    best_config, best_score = optimizer.get_best()
    print(f"\n   Mejor hasta ahora:")
    print(f"     Score: {best_score:.4f}")
    print(f"     Config: {best_config}")
    
    print("\n✅ Optimización funciona correctamente!")
    return True


def main():
    parser = argparse.ArgumentParser(description='Test pipeline flow')
    parser.add_argument('--algorithm', type=str, default='adaboost',
                       choices=['adaboost', 'random_forest', 'svc', 'mlp'])
    parser.add_argument('--epochs', type=int, default=100,
                       help='Épocas de entrenamiento (default: 100 para prueba)')
    parser.add_argument('--skip_prep', action='store_true',
                       help='Saltar preparación de datos')
    parser.add_argument('--skip_train', action='store_true',
                       help='Saltar entrenamiento')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🧪 TEST: Flujo Completo de FSBO Pipeline")
    print("=" * 60)
    print(f"\nAlgoritmo: {args.algorithm}")
    
    success = True
    
    # Paso 1: Preparación
    if not args.skip_prep:
        if not test_data_preparation(args.algorithm):
            print("\n❌ Falló la preparación de datos")
            success = False
    else:
        print("\n⏭️ Saltando preparación de datos")
    
    # Paso 2: Entrenamiento
    if success and not args.skip_train:
        if not test_training(args.algorithm, args.epochs):
            print("\n❌ Falló el entrenamiento")
            success = False
    elif args.skip_train:
        print("\n⏭️ Saltando entrenamiento")
    
    # Paso 3: Optimización
    if success:
        if not test_optimization(args.algorithm):
            print("\n❌ Falló la optimización")
            success = False
    
    # Resumen
    print("\n" + "=" * 60)
    if success:
        print("✅ TODOS LOS TESTS PASARON")
    else:
        print("❌ ALGUNOS TESTS FALLARON")
    print("=" * 60)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
