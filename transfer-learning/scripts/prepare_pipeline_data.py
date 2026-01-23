"""
Script para preparar los datos de pipeline completo para FSBO.

Lee los CSVs de pipes/combined y los transforma al formato necesario
para entrenar FSBO con pipeline completo (preprocesamiento + hiperparámetros).

Uso:
    python scripts/prepare_pipeline_data.py --algorithm adaboost
    python scripts/prepare_pipeline_data.py --algorithm all

Autor: Proyecto académico MetaLearning
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Mapeo de algoritmos: nombre CSV -> nombre interno
# =============================================================================

ALGORITHM_MAPPING = {
    'AdaBoostClassifier': 'adaboost',
    'BernoulliNB': 'bernoulli_nb',
    'DecisionTreeClassifier': 'decision_tree',
    'ExtraTreesClassifier': 'extra_trees',
    'GaussianNB': 'gaussian_nb',
    'HistGradientBoostingClassifier': 'hist_gradient_boosting',
    'KNeighborsClassifier': 'kneighbors',
    'LinearDiscriminantAnalysis': 'lda',
    'LinearSVC': 'linear_svc',
    'MLPClassifier': 'mlp',
    'MultinomialNB': 'multinomial_nb',
    'PassiveAggressiveClassifier': 'passive_aggressive',
    'QuadraticDiscriminantAnalysis': 'qda',
    'RandomForestClassifier': 'random_forest',
    'SGDClassifier': 'sgd',
    'SVC': 'svc',
}

# Columnas comunes del pipeline (one-hot ya aplicado)
PIPELINE_COLUMNS = [
    # Imputer Strategy
    'Imputer Strategy_none',
    'Imputer Strategy_simpleimputer',
    # Categorical Strategy
    'Categorical Strategy_none',
    'Categorical Strategy_onehot',
    'Categorical Strategy_ordinalencoder',
    # Feature Selection (14 opciones)
    'Feature Selection_extra_tree',
    'Feature Selection_fastica',
    'Feature Selection_feature_agglomeration',
    'Feature Selection_generic_univariate',
    'Feature Selection_kernel_pca',
    'Feature Selection_linear_svc',
    'Feature Selection_none',
    'Feature Selection_nystroem',
    'Feature Selection_pca',
    'Feature Selection_polynomial_features',
    'Feature Selection_random_trees_embedding',
    'Feature Selection_rbf_sampler',
    'Feature Selection_select_percentile',
    'Feature Selection_truncated_svd',
    # Scaler (7 opciones)
    'Scaler_minmax',
    'Scaler_none',
    'Scaler_normalizer',
    'Scaler_power',
    'Scaler_quantile',
    'Scaler_robust',
    'Scaler_standard',
]

# Columnas de metadata (no usar como features)
METADATA_COLUMNS = ['Dataset', 'Fold', 'Fold Accuracy', 'Training Time', 'Testing Time', 'random_state']

# Definición de hiperparámetros por algoritmo
ALGORITHM_HYPERPARAMS = {
    'adaboost': {
        'numeric': ['n_estimators', 'learning_rate', 'base_estimator__max_depth'],
        'categorical': [],
        'ranges': {
            'n_estimators': (10, 500),
            'learning_rate': (0.01, 10.0),
            'base_estimator__max_depth': (1, 10),
        },
        'log_scale': ['learning_rate'],
    },
    'svc': {
        'numeric': ['C', 'degree', 'coef0', 'tol'],
        'categorical': ['kernel', 'gamma', 'shrinking'],
        'ranges': {
            'C': (0.001, 100.0),
            'degree': (1, 5),
            'coef0': (-1.0, 1.0),
            'tol': (0.0001, 0.01),
        },
        'log_scale': ['C', 'tol'],
        'categorical_values': {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto'],
            'shrinking': ['True', 'False'],
        }
    },
    'random_forest': {
        'numeric': ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf'],
        'categorical': ['criterion', 'max_features', 'bootstrap'],
        'ranges': {
            'n_estimators': (10, 500),
            'max_depth': (1, 50),
            'min_samples_split': (2, 20),
            'min_samples_leaf': (1, 20),
        },
        'log_scale': [],
        'categorical_values': {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_features': ['sqrt', 'log2', ''],
            'bootstrap': ['True', 'False'],
        }
    },
    'decision_tree': {
        'numeric': ['max_depth', 'min_samples_split', 'min_samples_leaf'],
        'categorical': ['criterion', 'splitter', 'max_features'],
        'ranges': {
            'max_depth': (1, 50),
            'min_samples_split': (2, 20),
            'min_samples_leaf': (1, 20),
        },
        'log_scale': [],
        'categorical_values': {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'splitter': ['best', 'random'],
            'max_features': ['sqrt', 'log2', ''],
        }
    },
    'extra_trees': {
        'numeric': ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf'],
        'categorical': ['criterion', 'max_features', 'bootstrap'],
        'ranges': {
            'n_estimators': (10, 500),
            'max_depth': (1, 50),
            'min_samples_split': (2, 20),
            'min_samples_leaf': (1, 20),
        },
        'log_scale': [],
        'categorical_values': {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_features': ['sqrt', 'log2', ''],
            'bootstrap': ['True', 'False'],
        }
    },
    'kneighbors': {
        'numeric': ['n_neighbors', 'leaf_size', 'p'],
        'categorical': ['weights', 'algorithm', 'metric'],
        'ranges': {
            'n_neighbors': (1, 50),
            'leaf_size': (10, 100),
            'p': (1, 5),
        },
        'log_scale': [],
        'categorical_values': {
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
        }
    },
    'mlp': {
        'numeric': ['alpha', 'learning_rate_init', 'max_iter', 'hidden_layer_sizes'],
        'categorical': ['activation', 'solver', 'learning_rate'],
        'ranges': {
            'alpha': (0.0001, 1.0),
            'learning_rate_init': (0.001, 1.0),
            'max_iter': (100, 1000),
            'hidden_layer_sizes': (10, 200),  # Extraído de tupla, rango típico
        },
        'log_scale': ['alpha', 'learning_rate_init'],
        'categorical_values': {
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
        },
        'special_processing': ['hidden_layer_sizes'],  # Necesita extracción de tupla
    },
    'linear_svc': {
        'numeric': ['C', 'tol', 'max_iter'],
        'categorical': ['penalty', 'loss'],
        'ranges': {
            'C': (0.001, 100.0),
            'tol': (0.0001, 0.01),
            'max_iter': (100, 10000),
        },
        'log_scale': ['C', 'tol'],
        'categorical_values': {
            'penalty': ['l1', 'l2'],
            'loss': ['hinge', 'squared_hinge'],
        }
    },
    'sgd': {
        'numeric': ['alpha', 'l1_ratio', 'max_iter', 'tol', 'eta0'],
        'categorical': ['loss', 'penalty', 'learning_rate'],
        'ranges': {
            'alpha': (0.0001, 1.0),
            'l1_ratio': (0.0, 1.0),
            'max_iter': (100, 10000),
            'tol': (0.0001, 0.01),
            'eta0': (0.01, 1.0),
        },
        'log_scale': ['alpha', 'tol'],
        'categorical_values': {
            'loss': ['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron'],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
        }
    },
    'bernoulli_nb': {
        'numeric': ['alpha', 'binarize'],
        'categorical': ['fit_prior'],
        'ranges': {
            'alpha': (0.0, 10.0),
            'binarize': (0.0, 1.0),
        },
        'log_scale': [],
        'categorical_values': {
            'fit_prior': ['True', 'False'],
        }
    },
    'gaussian_nb': {
        'numeric': ['var_smoothing'],
        'categorical': [],
        'ranges': {
            'var_smoothing': (1e-12, 1e-6),
        },
        'log_scale': ['var_smoothing'],
    },
    'multinomial_nb': {
        'numeric': ['alpha'],
        'categorical': ['fit_prior'],
        'ranges': {
            'alpha': (0.0, 10.0),
        },
        'log_scale': [],
        'categorical_values': {
            'fit_prior': ['True', 'False'],
        }
    },
    'lda': {
        'numeric': ['tol'],
        'categorical': ['solver', 'shrinkage'],
        'ranges': {
            'tol': (0.0001, 0.01),
        },
        'log_scale': ['tol'],
        'categorical_values': {
            'solver': ['svd', 'lsqr', 'eigen'],
            'shrinkage': ['auto', 'None'],
        }
    },
    'qda': {
        'numeric': ['reg_param', 'tol'],
        'categorical': [],
        'ranges': {
            'reg_param': (0.0, 1.0),
            'tol': (0.0001, 0.01),
        },
        'log_scale': ['tol'],
    },
    'hist_gradient_boosting': {
        'numeric': ['learning_rate', 'max_iter', 'max_depth', 'min_samples_leaf', 'l2_regularization', 'max_bins'],
        'categorical': [],
        'ranges': {
            'learning_rate': (0.01, 1.0),
            'max_iter': (50, 500),
            'max_depth': (1, 50),
            'min_samples_leaf': (1, 100),
            'l2_regularization': (0.0, 10.0),
            'max_bins': (10, 255),
        },
        'log_scale': ['learning_rate'],
    },
    'passive_aggressive': {
        'numeric': ['C', 'tol', 'max_iter'],
        'categorical': ['loss', 'fit_intercept'],
        'ranges': {
            'C': (0.001, 10.0),
            'tol': (0.0001, 0.01),
            'max_iter': (100, 10000),
        },
        'log_scale': ['C', 'tol'],
        'categorical_values': {
            'loss': ['hinge', 'squared_hinge'],
            'fit_intercept': ['True', 'False'],
        }
    },
}


def extract_tuple_value(value: str) -> float:
    """Extrae el primer valor numérico de una tupla representada como string.
    
    Ejemplos:
        "(100,)" -> 100.0
        "(100, 50)" -> 100.0
        "100" -> 100.0
    """
    import re
    if pd.isna(value):
        return np.nan
    
    val_str = str(value).strip()
    # Buscar números en el string
    numbers = re.findall(r'\d+\.?\d*', val_str)
    if numbers:
        return float(numbers[0])
    return np.nan


def normalize_numeric(values: np.ndarray, low: float, high: float, log_scale: bool = False) -> np.ndarray:
    """Normaliza valores numéricos al rango [0, 1]."""
    values = np.array(values, dtype=float)
    
    # Manejar NaN
    mask = ~np.isnan(values)
    result = np.full_like(values, 0.5)  # Default para NaN
    
    if not mask.any():
        return result
    
    if log_scale and low > 0:
        values_clean = np.clip(values[mask], low, high)
        result[mask] = (np.log(values_clean) - np.log(low)) / (np.log(high) - np.log(low))
    else:
        result[mask] = (values[mask] - low) / (high - low)
    
    return np.clip(result, 0, 1)


def encode_categorical(values: pd.Series, categories: List[str]) -> np.ndarray:
    """Codifica valores categóricos como one-hot."""
    n_samples = len(values)
    n_categories = len(categories)
    one_hot = np.zeros((n_samples, n_categories), dtype=float)
    
    for i, val in enumerate(values):
        val_str = str(val).strip()
        if val_str in categories:
            idx = categories.index(val_str)
            one_hot[i, idx] = 1.0
        else:
            # Default: primera categoría
            one_hot[i, 0] = 1.0
    
    return one_hot


def process_algorithm_data(
    df: pd.DataFrame,
    algorithm: str,
    hp_config: Dict
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Procesa los datos de un algoritmo.
    
    Returns:
        X: Features (pipeline + hiperparámetros normalizados)
        y: Scores (accuracy)
        feature_names: Nombres de las columnas
    """
    feature_columns = []
    feature_data = []
    
    # 1. Añadir columnas del pipeline (ya están en one-hot)
    for col in PIPELINE_COLUMNS:
        if col in df.columns:
            feature_data.append(df[col].values.astype(float))
            feature_columns.append(col)
    
    # 2. Procesar hiperparámetros numéricos
    special_params = hp_config.get('special_processing', [])
    for hp in hp_config.get('numeric', []):
        if hp in df.columns:
            # Verificar si necesita procesamiento especial (ej. extraer de tupla)
            if hp in special_params:
                values = df[hp].apply(extract_tuple_value).values
            else:
                values = pd.to_numeric(df[hp], errors='coerce').values
            low, high = hp_config['ranges'].get(hp, (0, 1))
            log_scale = hp in hp_config.get('log_scale', [])
            normalized = normalize_numeric(values, low, high, log_scale)
            feature_data.append(normalized)
            feature_columns.append(f'hp_{hp}')
    
    # 3. Procesar hiperparámetros categóricos
    for hp in hp_config.get('categorical', []):
        if hp in df.columns:
            categories = hp_config.get('categorical_values', {}).get(hp, [])
            if categories:
                one_hot = encode_categorical(df[hp], categories)
                for i, cat in enumerate(categories):
                    feature_data.append(one_hot[:, i])
                    feature_columns.append(f'hp_{hp}_{cat}')
    
    # Combinar features
    X = np.column_stack(feature_data) if feature_data else np.zeros((len(df), 0))
    
    # Score
    y = df['Fold Accuracy'].values.astype(float)
    
    return X, y, feature_columns


def create_task_mapping(df: pd.DataFrame) -> Dict[str, int]:
    """Crea mapeo de Dataset (string) a task_id (int)."""
    datasets = df['Dataset'].unique()
    return {ds: i for i, ds in enumerate(sorted(datasets))}


def prepare_data_for_algorithm(
    input_path: Path,
    algorithm_name: str,
    output_dir: Path
) -> Optional[Dict]:
    """
    Prepara los datos de un algoritmo para FSBO.
    
    Args:
        input_path: Ruta al CSV de entrada
        algorithm_name: Nombre interno del algoritmo
        output_dir: Directorio de salida
        
    Returns:
        Información sobre los datos procesados
    """
    logger.info(f"Procesando {algorithm_name} desde {input_path.name}")
    
    # Verificar que tenemos configuración para este algoritmo
    if algorithm_name not in ALGORITHM_HYPERPARAMS:
        logger.warning(f"No hay configuración de hiperparámetros para {algorithm_name}, usando defaults")
        hp_config = {'numeric': [], 'categorical': [], 'ranges': {}, 'log_scale': []}
    else:
        hp_config = ALGORITHM_HYPERPARAMS[algorithm_name]
    
    # Leer datos
    df = pd.read_csv(input_path)
    logger.info(f"  Leídas {len(df)} filas")
    
    # Crear mapeo de tareas
    task_mapping = create_task_mapping(df)
    df['task_id'] = df['Dataset'].map(task_mapping)
    
    # Procesar features
    X, y, feature_names = process_algorithm_data(df, algorithm_name, hp_config)
    
    logger.info(f"  Features: {X.shape[1]} columnas")
    logger.info(f"  Tareas únicas: {len(task_mapping)}")
    
    # Crear DataFrame de salida
    output_df = pd.DataFrame(X, columns=feature_names)
    output_df.insert(0, 'task_id', df['task_id'].values)
    output_df['accuracy'] = y
    
    # Guardar CSV
    output_path = output_dir / f"{algorithm_name}_pipeline_representation.csv"
    output_df.to_csv(output_path, index=False)
    logger.info(f"  Guardado en {output_path.name}")
    
    # Guardar mapeo de tareas
    mapping_path = output_dir / f"{algorithm_name}_task_mapping.json"
    with open(mapping_path, 'w') as f:
        json.dump(task_mapping, f, indent=2)
    
    return {
        'algorithm': algorithm_name,
        'n_samples': len(df),
        'n_features': X.shape[1],
        'n_tasks': len(task_mapping),
        'feature_names': feature_names,
        'output_path': str(output_path),
    }


def generate_configspace(
    algorithm_name: str,
    feature_names: List[str],
    output_dir: Path
) -> None:
    """Genera el archivo configspace JSON para un algoritmo."""
    hp_config = ALGORITHM_HYPERPARAMS.get(algorithm_name, {})
    
    hyperparameters = []
    
    # Pipeline components (categóricos representados como grupos one-hot)
    # Imputer Strategy
    hyperparameters.append({
        "name": "pipeline__imputer_strategy",
        "type": "categorical",
        "choices": ["none", "simpleimputer"],
        "default": "simpleimputer"
    })
    
    # Categorical Strategy
    hyperparameters.append({
        "name": "pipeline__categorical_strategy",
        "type": "categorical",
        "choices": ["none", "onehot", "ordinalencoder"],
        "default": "onehot"
    })
    
    # Feature Selection
    hyperparameters.append({
        "name": "pipeline__feature_selection",
        "type": "categorical",
        "choices": [
            "extra_tree", "fastica", "feature_agglomeration", "generic_univariate",
            "kernel_pca", "linear_svc", "none", "nystroem", "pca",
            "polynomial_features", "random_trees_embedding", "rbf_sampler",
            "select_percentile", "truncated_svd"
        ],
        "default": "none"
    })
    
    # Scaler
    hyperparameters.append({
        "name": "pipeline__scaler",
        "type": "categorical",
        "choices": ["minmax", "none", "normalizer", "power", "quantile", "robust", "standard"],
        "default": "standard"
    })
    
    # Hiperparámetros numéricos del clasificador
    for hp in hp_config.get('numeric', []):
        low, high = hp_config['ranges'].get(hp, (0, 1))
        is_int = isinstance(low, int) and isinstance(high, int)
        
        hp_entry = {
            "name": f"classifier__{hp}",
            "type": "uniform_int" if is_int else "uniform_float",
            "lower": low,
            "upper": high,
            "log": hp in hp_config.get('log_scale', []),
            "default": low
        }
        hyperparameters.append(hp_entry)
    
    # Hiperparámetros categóricos del clasificador
    for hp in hp_config.get('categorical', []):
        categories = hp_config.get('categorical_values', {}).get(hp, [])
        if categories:
            hp_entry = {
                "name": f"classifier__{hp}",
                "type": "categorical",
                "choices": categories,
                "default": categories[0]
            }
            hyperparameters.append(hp_entry)
    
    configspace = {
        "name": f"Pipeline + {algorithm_name}",
        "hyperparameters": hyperparameters,
        "conditions": [],
        "forbiddens": []
    }
    
    output_path = output_dir / f"{algorithm_name}_pipeline_configspace.json"
    with open(output_path, 'w') as f:
        json.dump(configspace, f, indent=2)
    
    logger.info(f"  ConfigSpace guardado en {output_path.name}")


def main():
    parser = argparse.ArgumentParser(description='Preparar datos de pipeline para FSBO')
    parser.add_argument('--algorithm', type=str, default='all',
                       help='Algoritmo a procesar (o "all" para todos)')
    parser.add_argument('--input_dir', type=str, default=None,
                       help='Directorio de entrada (default: pipes/combined)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directorio de salida (default: transfer-learning/data/pipeline_representation)')
    
    args = parser.parse_args()
    
    # Rutas
    base_dir = Path(__file__).parent.parent.parent  # MetaLearning-
    
    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        input_dir = base_dir / 'pipes' / 'combined'
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = base_dir / 'transfer-learning' / 'data' / 'pipeline_representation'
    
    configspace_dir = base_dir / 'transfer-learning' / 'data' / 'pipeline_configspace'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    configspace_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("🔧 Preparación de Datos de Pipeline para FSBO")
    print("=" * 60)
    print(f"\nInput: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"ConfigSpace: {configspace_dir}")
    
    # Determinar algoritmos a procesar
    if args.algorithm == 'all':
        algorithms_to_process = list(ALGORITHM_MAPPING.items())
    else:
        # Buscar el algoritmo
        found = False
        for csv_name, internal_name in ALGORITHM_MAPPING.items():
            if internal_name == args.algorithm or csv_name.lower() == args.algorithm.lower():
                algorithms_to_process = [(csv_name, internal_name)]
                found = True
                break
        if not found:
            print(f"❌ Algoritmo no encontrado: {args.algorithm}")
            print(f"   Disponibles: {list(ALGORITHM_MAPPING.values())}")
            return
    
    results = []
    
    for csv_name, internal_name in algorithms_to_process:
        input_path = input_dir / f"{csv_name}_combined.csv"
        
        if not input_path.exists():
            logger.warning(f"No encontrado: {input_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"📊 {internal_name.upper()}")
        print(f"{'='*60}")
        
        result = prepare_data_for_algorithm(input_path, internal_name, output_dir)
        
        if result:
            results.append(result)
            generate_configspace(internal_name, result['feature_names'], configspace_dir)
    
    # Resumen
    print("\n" + "=" * 60)
    print("📋 RESUMEN")
    print("=" * 60)
    
    for r in results:
        print(f"\n✅ {r['algorithm']}:")
        print(f"   Muestras: {r['n_samples']}")
        print(f"   Features: {r['n_features']}")
        print(f"   Tareas: {r['n_tasks']}")
    
    print(f"\n📁 Datos guardados en: {output_dir}")
    print(f"📁 ConfigSpaces en: {configspace_dir}")
    print("\n¡Preparación completada!")


if __name__ == "__main__":
    main()
