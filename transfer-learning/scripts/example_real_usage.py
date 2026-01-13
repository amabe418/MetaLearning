"""
Ejemplo de uso del pipeline de producción con datos reales.

Este script demuestra cómo usar el sistema completo de meta-learning + FSBO
con datasets reales.
"""

import numpy as np
from production_pipeline import ProductionPipeline, ProductionConfig, optimize_dataset


def example_1_quick_usage():
    """Ejemplo 1: Uso rápido con función helper."""
    print("\n" + "=" * 70)
    print("📚 EJEMPLO 1: Uso Rápido")
    print("=" * 70)
    
    from sklearn.datasets import load_breast_cancer
    
    # Cargar dataset real
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    print(f"\n📊 Dataset: Breast Cancer")
    print(f"   Samples: {len(X)}")
    print(f"   Features: {X.shape[1]}")
    
    # Optimizar (función de una línea)
    result = optimize_dataset(
        X, y,
        dataset_id='breast_cancer',
        total_budget=50,
        max_algorithms=3,
        verbose=True
    )
    
    print(f"\n✅ Resultado:")
    print(f"   • Mejor algoritmo: {result.best_algorithm}")
    print(f"   • Score: {result.best_score:.4f}")
    
    return result


def example_2_custom_config():
    """Ejemplo 2: Configuración personalizada."""
    print("\n" + "=" * 70)
    print("📚 EJEMPLO 2: Configuración Personalizada")
    print("=" * 70)
    
    from sklearn.datasets import load_digits
    
    # Cargar dataset
    data = load_digits()
    X, y = data.data, data.target
    
    print(f"\n📊 Dataset: Digits")
    print(f"   Samples: {len(X)}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Classes: {len(np.unique(y))}")
    
    # Configuración personalizada
    config = ProductionConfig(
        total_budget=80,
        max_algorithms=4,
        similarity_threshold=0.4,  # Menos restrictivo
        min_similar_tasks=1,
        n_warm_start=7,
        early_stopping_patience=8,
        verbose=True
    )
    
    # Crear y ejecutar pipeline
    pipeline = ProductionPipeline(config)
    result = pipeline.run(
        X, y,
        dataset_id='digits',
        test_size=0.2
    )
    
    print(f"\n✅ Resultado:")
    print(f"   • Algoritmo: {result.best_algorithm}")
    print(f"   • Score: {result.best_score:.4f}")
    print(f"   • Evaluaciones: {result.total_evaluations}")
    
    return result


def example_3_custom_evaluation():
    """Ejemplo 3: Función de evaluación personalizada."""
    print("\n" + "=" * 70)
    print("📚 EJEMPLO 3: Evaluación Personalizada")
    print("=" * 70)
    
    from sklearn.datasets import load_wine
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score
    
    # Cargar dataset
    data = load_wine()
    X, y = data.data, data.target
    
    print(f"\n📊 Dataset: Wine")
    print(f"   Samples: {len(X)}")
    
    # Función de evaluación personalizada
    def custom_evaluation(algorithm, config, X_train, y_train, X_val, y_val):
        """Evaluación con preprocesamiento y métrica F1."""
        
        # Preprocesar
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Limpiar config
        clean_config = {
            k.split('__')[-1]: v
            for k, v in config.items()
            if not k.startswith('imputation')
        }
        
        # Crear modelo
        if algorithm == 'random_forest':
            model = RandomForestClassifier(**clean_config)
        elif algorithm == 'adaboost':
            model = AdaBoostClassifier(**clean_config)
        else:
            return 0.5
        
        try:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_val_scaled)
            
            # Usar F1-score en lugar de accuracy
            score = f1_score(y_val, y_pred, average='weighted')
            return score
            
        except Exception as e:
            print(f"   ⚠️ Error en evaluación: {e}")
            return 0.0
    
    # Ejecutar con evaluación personalizada
    config = ProductionConfig(
        total_budget=60,
        max_algorithms=2,
        verbose=True
    )
    
    pipeline = ProductionPipeline(config)
    result = pipeline.run(
        X, y,
        dataset_id='wine',
        evaluation_fn=custom_evaluation,
        save_to_kb=True
    )
    
    print(f"\n✅ Resultado (F1-score):")
    print(f"   • Algoritmo: {result.best_algorithm}")
    print(f"   • F1-score: {result.best_score:.4f}")
    
    return result


def example_4_analyzing_kb():
    """Ejemplo 4: Analizar la knowledge base."""
    print("\n" + "=" * 70)
    print("📚 EJEMPLO 4: Análisis de Knowledge Base")
    print("=" * 70)
    
    from pipeline import KnowledgeBase
    from meta_learner import analyze_knowledge_base
    
    # Cargar KB
    kb = KnowledgeBase('experiments/knowledge_base.json')
    
    # Analizar
    stats = analyze_knowledge_base(kb)
    
    print(f"\n📊 Estadísticas de Knowledge Base:")
    print(f"   • Total de entradas: {stats['total_entries']}")
    print(f"   • Datasets únicos: {stats['datasets']}")
    
    print(f"\n🤖 Por Algoritmo:")
    for algo, data in stats['algorithms'].items():
        print(f"   • {algo}:")
        print(f"      - Entradas: {data['count']}")
        print(f"      - Score promedio: {data['mean_score']:.4f}")
        print(f"      - Rango: [{data['min_score']:.4f}, {data['max_score']:.4f}]")


def example_5_openml_dataset():
    """Ejemplo 5: Usar dataset de OpenML."""
    print("\n" + "=" * 70)
    print("📚 EJEMPLO 5: Dataset de OpenML")
    print("=" * 70)
    
    try:
        from sklearn.datasets import fetch_openml
        
        print("\n📥 Descargando dataset de OpenML...")
        
        # Cargar dataset real de OpenML
        # Puedes cambiar por cualquier dataset de OpenML
        data = fetch_openml('credit-g', version=1, parser='auto')
        X = data.data.to_numpy() if hasattr(data.data, 'to_numpy') else data.data
        y = data.target.to_numpy() if hasattr(data.target, 'to_numpy') else data.target
        
        # Convertir labels a numéricos si es necesario
        if y.dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        print(f"\n📊 Dataset: Credit-G (OpenML)")
        print(f"   Samples: {len(X)}")
        print(f"   Features: {X.shape[1]}")
        
        # Optimizar
        result = optimize_dataset(
            X, y,
            dataset_id='credit-g',
            total_budget=80,
            max_algorithms=3,
            verbose=True
        )
        
        print(f"\n✅ Resultado:")
        print(f"   • Algoritmo: {result.best_algorithm}")
        print(f"   • Score: {result.best_score:.4f}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error cargando de OpenML: {e}")
        print("   Asegúrate de tener conexión a internet")


def main():
    """Ejecuta todos los ejemplos."""
    print("\n" + "=" * 70)
    print("🚀 EJEMPLOS DE USO - PRODUCTION PIPELINE")
    print("=" * 70)
    
    # Ejecutar ejemplos
    examples = [
        ("Uso Rápido", example_1_quick_usage),
        ("Config Personalizada", example_2_custom_config),
        ("Evaluación Custom", example_3_custom_evaluation),
        ("Análisis KB", example_4_analyzing_kb),
        # ("OpenML", example_5_openml_dataset),  # Comentado para evitar download
    ]
    
    results = []
    for name, example_fn in examples:
        try:
            result = example_fn()
            results.append((name, result))
            print("\n" + "✅" * 35 + "\n")
        except Exception as e:
            print(f"\n❌ Error en {name}: {e}\n")
    
    # Resumen final
    print("\n" + "=" * 70)
    print("📊 RESUMEN DE TODOS LOS EJEMPLOS")
    print("=" * 70)
    
    for name, result in results:
        if result:
            print(f"\n{name}:")
            print(f"   • Algoritmo: {result.best_algorithm}")
            print(f"   • Score: {result.best_score:.4f}")


if __name__ == "__main__":
    # Ejecutar un ejemplo específico
    import sys
    
    if len(sys.argv) > 1:
        example_num = int(sys.argv[1])
        examples = [
            example_1_quick_usage,
            example_2_custom_config,
            example_3_custom_evaluation,
            example_4_analyzing_kb,
            example_5_openml_dataset,
        ]
        
        if 1 <= example_num <= len(examples):
            examples[example_num - 1]()
        else:
            print(f"Ejemplo {example_num} no existe. Usa 1-{len(examples)}")
    else:
        # Ejecutar todos
        main()

