"""
Script de Prueba para metafeatx_fsbo_integration.py

Este script prueba la integración completa MetaFeatX + FSBO con datasets REALES.

Fuentes de datasets:
1. OpenML (recomendado): https://www.openml.org/
2. sklearn: load_iris, load_wine, load_breast_cancer, etc.

Requisitos:
- Datos históricos en ./data/:
  - basic_representations.csv
  - {algoritmo}_target_representation.csv (para cada algoritmo)
  - top_raw_target_representation/{algoritmo}_target_representation.csv
- Modelos FSBO pre-entrenados (opcional, si no están, usará random search)
- pymfe instalado: pip install pymfe
- openml instalado: pip install openml (para datasets de OpenML)

Uso:
    python test_metafeatx_fsbo_integration.py
"""

import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import sys

# Agregar path del proyecto
sys.path.insert(0, str(Path(__file__).parent))

from metafeatx_fsbo_integration import integrate_metafeatx_fsbo, MetaFeatXFSBOIntegration


def create_evaluation_function(X_train, y_train, X_val, y_val):
    """
    Crea una función de evaluación para los algoritmos.
    
    Esta función entrena un modelo con la configuración dada y retorna el score.
    """
    def evaluate(algorithm: str, config: dict) -> float:
        """Evalúa un algoritmo con una configuración dada."""
        models = {
            'random_forest': RandomForestClassifier,
            'adaboost': AdaBoostClassifier,
            'libsvm_svc': SVC,
        }
        
        # Limpiar config (remover prefijos si existen)
        clean_config = {
            k.split('__')[-1]: v
            for k, v in config.items()
            if not k.startswith('imputation')
        }
        
        if algorithm not in models:
            print(f"⚠️  Algoritmo {algorithm} no soportado, retornando score 0.5")
            return 0.5
        
        try:
            model = models[algorithm](**clean_config, random_state=42)
            model.fit(X_train, y_train)
            score = float(model.score(X_val, y_val))
            return score
        except Exception as e:
            print(f"❌ Error evaluando {algorithm} con config {clean_config}: {e}")
            return 0.0
    
    return evaluate


def load_openml_dataset(dataset_id, dataset_name=None):
    """
    Carga un dataset de OpenML.
    
    Args:
        dataset_id: ID del dataset en OpenML (ej: 61 para iris, 37 para diabetes)
        dataset_name: Nombre del dataset (opcional, para mostrar)
    
    Returns:
        X, y: Features y target como arrays numpy
    """
    try:
        import openml
        
        print(f"📥 Descargando dataset de OpenML (ID: {dataset_id})...")
        dataset = openml.datasets.get_dataset(dataset_id)
        
        if dataset_name is None:
            dataset_name = dataset.name
        
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        
        # Convertir a numpy arrays
        if hasattr(X, 'to_numpy'):
            X = X.to_numpy()
        if hasattr(y, 'to_numpy'):
            y = y.to_numpy()
        
        # Convertir labels categóricos a numéricos
        if y.dtype == 'object' or y.dtype.name == 'category':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        # Manejar valores faltantes (simple imputación)
        from sklearn.impute import SimpleImputer
        if np.isnan(X).any():
            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(X)
        
        print(f"   ✓ Dataset cargado: {dataset_name}")
        return X, y, dataset_name
        
    except ImportError:
        raise ImportError("openml no está instalado. Instala con: pip install openml")
    except Exception as e:
        raise Exception(f"Error cargando dataset de OpenML {dataset_id}: {e}")


def test_with_openml_dataset(dataset_id, dataset_name=None):
    """Prueba con un dataset real de OpenML."""
    print("=" * 70)
    print(f"🧪 PRUEBA 1: Dataset OpenML (ID: {dataset_id})")
    print("=" * 70)
    
    try:
        # Cargar dataset de OpenML
        X, y, name = load_openml_dataset(dataset_id, dataset_name)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"\n📊 Dataset: {name}")
        print(f"   Train: {X_train.shape}")
        print(f"   Val: {X_val.shape}")
        print(f"   Classes: {len(np.unique(y))}")
        
        # Crear función de evaluación
        evaluate = create_evaluation_function(X_train, y_train, X_val, y_val)
        
        # Ejecutar integración
        result = integrate_metafeatx_fsbo(
            X=X,
            y=y,
            evaluation_fn=evaluate,
            data_path="./data",
            conf_path="./conf",
            top_k_algorithms=3,
            budget_per_algorithm=20,  # Reducido para prueba rápida
            verbose=True
        )
        
        print("\n" + "=" * 70)
        print("✅ PRUEBA 1 COMPLETADA")
        print("=" * 70)
        print(f"\n📊 Resultados:")
        print(f"   Dataset: {name}")
        print(f"   Mejor algoritmo: {result.best_algorithm}")
        print(f"   Mejor score: {result.best_score:.4f}")
        print(f"   Total evaluaciones: {result.total_evaluations}")
        print(f"\n   Algoritmos sugeridos:")
        for algo in result.suggested_algorithms:
            res = result.optimization_results[algo]
            print(f"     • {algo}: {res.best_score:.4f} ({res.n_evaluations} evals)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error en prueba 1: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_sklearn_dataset(loader_func, dataset_name):
    """Prueba con un dataset de sklearn."""
    print("\n" + "=" * 70)
    print(f"🧪 PRUEBA 2: Dataset {dataset_name} (sklearn)")
    print("=" * 70)
    
    # Cargar dataset de sklearn
    data = loader_func(return_X_y=True)
    X, y = data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n📊 Dataset {dataset_name}:")
    print(f"   Train: {X_train.shape}")
    print(f"   Val: {X_val.shape}")
    print(f"   Classes: {len(np.unique(y))}")
    
    # Crear función de evaluación
    evaluate = create_evaluation_function(X_train, y_train, X_val, y_val)
    
    # Ejecutar integración
    try:
        result = integrate_metafeatx_fsbo(
            X=X,
            y=y,
            evaluation_fn=evaluate,
            data_path="./data",
            conf_path="./conf",
            top_k_algorithms=2,  # Solo 2 para prueba más rápida
            budget_per_algorithm=15,
            verbose=True
        )
        
        print("\n" + "=" * 70)
        print("✅ PRUEBA 2 COMPLETADA")
        print("=" * 70)
        print(f"\n📊 Resultados:")
        print(f"   Dataset: {dataset_name}")
        print(f"   Mejor algoritmo: {result.best_algorithm}")
        print(f"   Mejor score: {result.best_score:.4f}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error en prueba 2: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_wine():
    """Prueba con el dataset Wine de sklearn."""
    return test_with_sklearn_dataset(load_wine, "Wine")


def test_with_breast_cancer():
    """Prueba con el dataset Breast Cancer de sklearn."""
    return test_with_sklearn_dataset(load_breast_cancer, "Breast Cancer")


def main():
    """Ejecuta todas las pruebas."""
    print("\n" + "=" * 70)
    print("🚀 TESTING: Integración MetaFeatX + FSBO (con Datasets REALES)")
    print("=" * 70)
    print("\nEste script prueba la integración completa con datasets reales.")
    print("Asegúrate de tener:")
    print("  ✓ Datos históricos en ./data/")
    print("  ✓ pymfe instalado (pip install pymfe)")
    print("  ✓ openml instalado (pip install openml) - para datasets de OpenML")
    print("  ✓ Modelos FSBO pre-entrenados (opcional)\n")
    
    results = []
    
    # Prueba 1: Dataset de OpenML (ejemplo: iris, ID=61)
    # Puedes cambiar el ID por cualquier dataset de OpenML
    # Algunos IDs populares:
    # - 61: iris
    # - 37: diabetes
    # - 151: credit-g
    # - 1461: bank-marketing
    # - 40981: credit-card-fraud
    try:
        results.append(("OpenML Dataset (Iris, ID=61)", test_with_openml_dataset(61, "Iris")))
    except Exception as e:
        print(f"⚠️  Saltando prueba de OpenML: {e}")
        print("   Instala openml: pip install openml")
    
    # Prueba 2: Dataset de sklearn (Iris)
    results.append(("Iris (sklearn)", test_with_sklearn_dataset(load_iris, "Iris")))
    
    # Prueba 3: Dataset de sklearn (Wine)
    results.append(("Wine (sklearn)", test_with_wine()))
    
    # Prueba 4: Dataset de sklearn (Breast Cancer) - opcional, más grande
    # Descomenta si quieres probar con un dataset más grande:
    # results.append(("Breast Cancer (sklearn)", test_with_breast_cancer()))
    
    # Resumen
    print("\n" + "=" * 70)
    print("📋 RESUMEN DE PRUEBAS")
    print("=" * 70)
    
    for name, success in results:
        status = "✅ PASÓ" if success else "❌ FALLÓ"
        print(f"  {status}: {name}")
    
    total = len(results)
    passed = sum(1 for _, success in results if success)
    
    print(f"\n  Total: {total} pruebas")
    print(f"  Pasaron: {passed}")
    print(f"  Fallaron: {total - passed}")
    
    if passed == total:
        print("\n🎉 ¡Todas las pruebas pasaron!")
    else:
        print("\n⚠️  Algunas pruebas fallaron. Revisa los errores arriba.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
