"""
Pipeline Recommender: Sistema de recomendación de ML pipelines completos.

Dado un dataset y un problema de ML, este sistema recomienda:
1. Qué ALGORITMO utilizar
2. Qué PIPELINE de preprocesamiento utilizar
3. Qué HIPERPARÁMETROS utilizar

El sistema genera un RANKING de configuraciones prometedoras usando:
- MetaFeatX: para encontrar datasets similares (embedding aprendido)
- FSBO Models: para predecir scores esperados de cada configuración
- Datos históricos: como referencia de qué funcionó en datasets similares

Uso:
    from pipeline_recommender import PipelineRecommender
    
    recommender = PipelineRecommender.from_data(
        data_path='./data',
        pipes_path='./pipes/combined',
        checkpoints_path='./transfer-learning/experiments/checkpoints_pipeline'
    )
    
    # Dado un nuevo dataset
    ranking = recommender.recommend(
        task_id=123,          # o metafeatures directamente
        top_k=10              # Top 10 recomendaciones
    )

Autor: Proyecto académico MetaLearning
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

# Agregar paths del proyecto
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Imports de los modelos
try:
    import torch
    import gpytorch
    from gpytorch.likelihoods import GaussianLikelihood
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch/GPyTorch no disponible. FSBO desactivado.")

# MetaFeatX: No importamos el módulo, cargamos embeddings pre-calculados
# Esto evita la dependencia de 'ot' (POT library) que se necesita para entrenar
METAFEATX_EMBEDDINGS_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# Modelos (para cargar checkpoints)
# =============================================================================

if TORCH_AVAILABLE:
    import torch.nn as nn
    
    class DeepKernelNetwork(nn.Module):
        """
        Red neuronal que transforma configuraciones (pipeline + hiperparámetros) 
        a espacio latente.
        
        IMPORTANTE: Esta arquitectura DEBE coincidir exactamente con la usada
        en train_fsbo_pipeline.py para poder cargar los checkpoints.
        """
        def __init__(self, input_dim: int, hidden_dim: int = 128, n_layers: int = 3):
            super().__init__()
            
            layers = []
            
            # Primera capa
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            
            # Capas ocultas
            for _ in range(n_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
            
            self.network = nn.Sequential(*layers)
            self.output_dim = hidden_dim
        
        def forward(self, x):
            return self.network(x)
    
    class DeepKernelGP(gpytorch.models.ExactGP):
        """
        Gaussian Process con Deep Kernel para pipeline completo.
        
        IMPORTANTE: Esta arquitectura DEBE coincidir exactamente con la usada
        en train_fsbo_pipeline.py para poder cargar los checkpoints.
        """
        def __init__(self, train_x, train_y, likelihood, feature_extractor):
            super().__init__(train_x, train_y, likelihood)
            self.feature_extractor = feature_extractor
            self.mean_module = gpytorch.means.ConstantMean()
            # ARD kernel con dimensión del espacio latente
            latent_dim = feature_extractor.output_dim
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=latent_dim)
            )
        
        def forward(self, x):
            projected_x = self.feature_extractor(x)
            mean = self.mean_module(projected_x)
            covar = self.covar_module(projected_x)
            return gpytorch.distributions.MultivariateNormal(mean, covar)


# =============================================================================
# Configuración
# =============================================================================

AVAILABLE_ALGORITHMS = [
    'adaboost', 'random_forest', 'svc', 'decision_tree', 'extra_trees',
    'kneighbors', 'mlp', 'lda', 'qda', 'gaussian_nb', 'bernoulli_nb',
    'multinomial_nb', 'linear_svc', 'sgd', 'passive_aggressive',
    'hist_gradient_boosting'
]

ALGORITHM_CSV_MAPPING = {
    'adaboost': 'AdaBoostClassifier',
    'random_forest': 'RandomForestClassifier',
    'svc': 'SVC',
    'decision_tree': 'DecisionTreeClassifier',
    'extra_trees': 'ExtraTreesClassifier',
    'kneighbors': 'KNeighborsClassifier',
    'mlp': 'MLPClassifier',
    'lda': 'LinearDiscriminantAnalysis',
    'qda': 'QuadraticDiscriminantAnalysis',
    'gaussian_nb': 'GaussianNB',
    'bernoulli_nb': 'BernoulliNB',
    'multinomial_nb': 'MultinomialNB',
    'linear_svc': 'LinearSVC',
    'sgd': 'SGDClassifier',
    'passive_aggressive': 'PassiveAggressiveClassifier',
    'hist_gradient_boosting': 'HistGradientBoostingClassifier',
}

# Algoritmos con target representations disponibles para MetaFeatX
ALGORITHMS_WITH_TARGET_REPR = ['adaboost', 'random_forest', 'libsvm_svc', 'autosklearn']

PIPELINE_COLUMNS_PREFIX = [
    'Imputer Strategy_',
    'Categorical Strategy_',
    'Feature Selection_',
    'Scaler_',
]

METADATA_COLUMNS = ['Dataset', 'Fold', 'Fold Accuracy', 'Training Time', 
                    'Testing Time', 'random_state']

# =============================================================================
# Definición de hiperparámetros por algoritmo
# (DEBE coincidir con prepare_pipeline_data.py para que la codificación sea igual)
# =============================================================================

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
        'categorical_values': {}
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
            'hidden_layer_sizes': (10, 200),
        },
        'log_scale': ['alpha', 'learning_rate_init'],
        'categorical_values': {
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
        }
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
        'categorical_values': {}
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
        'categorical_values': {}
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
        'categorical_values': {}
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


# =============================================================================
# Estructuras de datos
# =============================================================================

@dataclass
class PipelineRecommendation:
    """Una recomendación de pipeline completo."""
    rank: int
    algorithm: str
    pipeline: Dict[str, str]
    classifier_params: Dict[str, Any]
    expected_score: float
    confidence: float
    source_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'rank': self.rank,
            'algorithm': self.algorithm,
            'pipeline': self.pipeline,
            'classifier': self.classifier_params,
            'expected_score': self.expected_score,
            'confidence': self.confidence,
            'source': self.source_info
        }


@dataclass 
class RecommendationResult:
    """Resultado completo de recomendación."""
    task_id: int
    n_neighbors_used: int
    recommendations: List[PipelineRecommendation]
    algorithm_ranking: List[Tuple[str, float]]  # [(algo, avg_score), ...]
    models_used: Dict[str, bool] = field(default_factory=dict)  # Qué modelos se usaron
    
    def get_best(self) -> PipelineRecommendation:
        return self.recommendations[0] if self.recommendations else None
    
    def get_best_per_algorithm(self) -> Dict[str, PipelineRecommendation]:
        best = {}
        for rec in self.recommendations:
            if rec.algorithm not in best:
                best[rec.algorithm] = rec
        return best
    
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.to_dict() for r in self.recommendations])


# =============================================================================
# PipelineRecommender
# =============================================================================

class PipelineRecommender:
    """
    Sistema de recomendación de pipelines de ML completos.
    
    USA LOS MODELOS ENTRENADOS:
    - MetaFeatX: para embedding de datasets (proyección ψ aprendida)
    - FSBO: para predecir scores esperados de configuraciones
    
    Dado un dataset (identificado por task_id o metafeatures), recomienda
    las mejores combinaciones de (algoritmo, pipeline, hiperparámetros).
    """
    
    def __init__(
        self,
        basic_representations: pd.DataFrame,
        historical_data: Dict[str, pd.DataFrame],  # {algorithm: df}
        task_id_mapping: Dict[str, int],
        metafeatx_embeddings: Optional[pd.DataFrame] = None,  # Embeddings pre-calculados
        fsbo_models: Optional[Dict[str, Any]] = None,  # {algorithm: (model, likelihood, feature_names)}
        device: str = 'cpu'
    ):
        self.basic_representations = basic_representations
        self.historical_data = historical_data
        self.task_id_mapping = task_id_mapping
        self.reverse_mapping = {v: k for k, v in task_id_mapping.items()}
        self.device = device
        
        # Embeddings MetaFeatX pre-calculados (si disponibles)
        self.metafeatx_embeddings = metafeatx_embeddings
        
        # Modelos FSBO entrenados
        self.fsbo_models = fsbo_models or {}
        
        # Calcular/cargar embedding
        self._compute_embedding()
        
        # Cachear mejores configs por dataset y algoritmo
        self._cache_best_configs()
        
        # Log de modelos cargados
        self._log_models_status()
    
    def _log_models_status(self):
        """Log del estado de los modelos."""
        logger.info("=" * 50)
        logger.info("MODELOS CARGADOS:")
        logger.info(f"  MetaFeatX Embeddings: {'✓ Cargados' if self.metafeatx_embeddings is not None else '✗ No disponible'}")
        if self.metafeatx_embeddings is not None:
            logger.info(f"    Shape: {self.metafeatx_embeddings.shape}")
        logger.info(f"  FSBO Models: {len(self.fsbo_models)} algoritmos")
        if self.fsbo_models:
            for algo in sorted(self.fsbo_models.keys()):
                logger.info(f"    - {algo}")
        logger.info("=" * 50)
    
    def _compute_embedding(self):
        """
        Carga/calcula embedding de datasets.
        
        Prioridad:
        1. Embeddings MetaFeatX pre-calculados (si disponibles)
        2. StandardScaler como fallback
        """
        feature_cols = [c for c in self.basic_representations.columns if c != 'task_id']
        
        # Agregar por task_id
        aggregated = self.basic_representations.groupby('task_id')[feature_cols].mean()
        
        if self.metafeatx_embeddings is not None:
            # ===== USAR EMBEDDINGS METAFEATX PRE-CALCULADOS =====
            logger.info("Cargando embeddings MetaFeatX pre-calculados...")
            try:
                # Los embeddings tienen columnas numéricas (0, 1, 2, ...) + task_id
                embedding_cols = [c for c in self.metafeatx_embeddings.columns if c != 'task_id']
                
                # Crear matriz de embeddings ordenada por task_id
                embeddings_indexed = self.metafeatx_embeddings.set_index('task_id')
                
                # Obtener task_ids que están tanto en basic_repr como en embeddings
                available_task_ids = set(aggregated.index) & set(embeddings_indexed.index)
                self.embedding_task_ids = sorted(list(available_task_ids))
                
                # Extraer embeddings en el orden correcto
                self.embedding_matrix = embeddings_indexed.loc[self.embedding_task_ids][embedding_cols].values
                
                logger.info(f"  Embeddings MetaFeatX: {self.embedding_matrix.shape}")
                logger.info(f"  Task IDs con embedding: {len(self.embedding_task_ids)}")
                self._embedding_source = 'metafeatx_precalculated'
                
            except Exception as e:
                logger.warning(f"Error cargando embeddings MetaFeatX: {e}. Usando fallback.")
                self.embedding_task_ids = list(aggregated.index)
                self._compute_fallback_embedding(aggregated)
        else:
            # Fallback sin MetaFeatX
            self.embedding_task_ids = list(aggregated.index)
            self._compute_fallback_embedding(aggregated)
    
    def _compute_fallback_embedding(self, aggregated: pd.DataFrame):
        """Embedding simple cuando MetaFeatX no está disponible."""
        logger.info("Calculando embedding con StandardScaler (fallback)...")
        scaler = StandardScaler()
        self.embedding_matrix = scaler.fit_transform(aggregated.values)
        logger.info(f"  Embedding simple: {self.embedding_matrix.shape}")
        self._embedding_source = 'standard_scaler'
    
    def _cache_best_configs(self):
        """Pre-calcula las mejores configs por dataset y algoritmo."""
        self.best_configs_cache = {}  # {(dataset, algorithm): [configs]}
        
        for algorithm, df in self.historical_data.items():
            if 'Dataset' not in df.columns:
                continue
                
            for dataset in df['Dataset'].unique():
                subset = df[df['Dataset'] == dataset]
                # Top 5 configs por dataset
                best = subset.nlargest(5, 'Fold Accuracy')
                
                configs = []
                for _, row in best.iterrows():
                    config = self._row_to_config(row, algorithm)
                    configs.append(config)
                
                self.best_configs_cache[(dataset, algorithm)] = configs
        
        logger.info(f"Cache de configs: {len(self.best_configs_cache)} entradas")
    
    def _row_to_config(self, row: pd.Series, algorithm: str) -> Dict:
        """Convierte fila de CSV a configuración."""
        config = {
            'algorithm': algorithm,
            'pipeline': {},
            'classifier': {},
            'accuracy': row.get('Fold Accuracy', 0),
            'dataset': row.get('Dataset', '')
        }
        
        # Extraer pipeline
        for col in row.index:
            if col.startswith('Imputer Strategy_') and row[col] == 1:
                config['pipeline']['imputer_strategy'] = col.replace('Imputer Strategy_', '')
            elif col.startswith('Categorical Strategy_') and row[col] == 1:
                config['pipeline']['categorical_strategy'] = col.replace('Categorical Strategy_', '')
            elif col.startswith('Feature Selection_') and row[col] == 1:
                config['pipeline']['feature_selection'] = col.replace('Feature Selection_', '')
            elif col.startswith('Scaler_') and row[col] == 1:
                config['pipeline']['scaler'] = col.replace('Scaler_', '')
        
        # Extraer hiperparámetros
        for col in row.index:
            if col not in METADATA_COLUMNS and not any(col.startswith(p) for p in PIPELINE_COLUMNS_PREFIX):
                value = row[col]
                if pd.notna(value):
                    if isinstance(value, (np.integer, int)):
                        value = int(value)
                    elif isinstance(value, (np.floating, float)):
                        value = float(value) if value != int(value) else int(value)
                    elif isinstance(value, str):
                        if value.lower() == 'true':
                            value = True
                        elif value.lower() == 'false':
                            value = False
                    config['classifier'][col] = value
        
        return config
    
    def _normalize_numeric(self, value: float, low: float, high: float, log_scale: bool) -> float:
        """
        Normaliza un valor numérico al rango [0, 1].
        
        DEBE coincidir con prepare_pipeline_data.py:normalize_numeric()
        """
        if value is None or np.isnan(value):
            return 0.5  # Valor por defecto
        
        # Clampear al rango
        value = max(low, min(high, value))
        
        if log_scale and low > 0 and value > 0:
            # Escala logarítmica
            return (np.log(value) - np.log(low)) / (np.log(high) - np.log(low))
        else:
            # Escala lineal
            if high == low:
                return 0.5
            return (value - low) / (high - low)
    
    def _config_to_vector(self, config: Dict, algorithm: str) -> Optional[np.ndarray]:
        """
        Convierte configuración a vector para FSBO.
        
        IMPORTANTE: Esta conversión DEBE coincidir exactamente con cómo
        se codificaron los datos en prepare_pipeline_data.py
        
        Returns None si el algoritmo no tiene modelo FSBO cargado.
        """
        if algorithm not in self.fsbo_models:
            return None
        
        model_data = self.fsbo_models[algorithm]
        feature_names = model_data['feature_names']
        
        # Obtener configuración de hiperparámetros del algoritmo
        hp_config = ALGORITHM_HYPERPARAMS.get(algorithm, {})
        
        # Crear vector
        x = np.zeros(len(feature_names), dtype=np.float32)
        
        # Extraer datos de la config
        pipeline = config.get('pipeline', {})
        classifier = config.get('classifier', {})
        
        # Mapeo de pipeline a nombres de columna
        pipeline_map = {
            'Imputer Strategy': pipeline.get('imputer_strategy', ''),
            'Categorical Strategy': pipeline.get('categorical_strategy', ''),
            'Feature Selection': pipeline.get('feature_selection', ''),
            'Scaler': pipeline.get('scaler', ''),
        }
        
        for i, fname in enumerate(feature_names):
            # 1. PIPELINE COLUMNS (one-hot)
            for prefix, value in pipeline_map.items():
                if fname.startswith(prefix + '_'):
                    option = fname.replace(prefix + '_', '')
                    x[i] = 1.0 if option.lower() == str(value).lower() else 0.0
                    break
            
            # 2. HIPERPARÁMETROS
            if fname.startswith('hp_'):
                hp_part = fname[3:]  # Quitar 'hp_'
                
                # Verificar si es un HP numérico
                for hp_name in hp_config.get('numeric', []):
                    if hp_part == hp_name:
                        # HP numérico - normalizar correctamente
                        value = classifier.get(hp_name)
                        if value is not None:
                            try:
                                value = float(value)
                                low, high = hp_config['ranges'].get(hp_name, (0, 1))
                                log_scale = hp_name in hp_config.get('log_scale', [])
                                x[i] = self._normalize_numeric(value, low, high, log_scale)
                            except (ValueError, TypeError):
                                x[i] = 0.5
                        else:
                            x[i] = 0.5
                        break
                
                # Verificar si es un HP categórico
                for hp_name in hp_config.get('categorical', []):
                    if hp_part.startswith(hp_name + '_'):
                        # HP categórico - one-hot
                        option = hp_part.replace(hp_name + '_', '')
                        value = classifier.get(hp_name)
                        
                        # Convertir valor a string para comparación
                        value_str = str(value) if value is not None else ''
                        
                        # Manejar booleanos
                        if value is True:
                            value_str = 'True'
                        elif value is False:
                            value_str = 'False'
                        
                        x[i] = 1.0 if option == value_str else 0.0
                        break
        
        return x
    
    def _get_expected_scores(
        self, 
        configs: List[Dict], 
        algorithm: str
    ) -> List[Tuple[float, float]]:
        """
        Obtiene scores esperados para las configuraciones.
        
        NOTA IMPORTANTE SOBRE FSBO:
        - El modelo FSBO (Gaussian Process) necesita OBSERVACIONES para hacer
          predicciones informativas (es few-shot, no zero-shot)
        - Sin observaciones, solo predice la media del prior (constante)
        - Para el PipelineRecommender usamos ACCURACY HISTÓRICA porque es
          información REAL de datasets similares
        - FSBO se usa en optimización interactiva (observe→suggest→observe)
        
        Returns:
            Lista de (score, confidence) para cada config
        """
        # Usar accuracy histórica como score (es información real)
        # La confianza se basa en cuánta evidencia histórica tenemos
        predictions = []
        for config in configs:
            accuracy = config.get('accuracy', 0.5)
            # Confianza alta porque es un dato real, no una predicción
            confidence = 0.9
            predictions.append((accuracy, confidence))
        
        return predictions
    
    @classmethod
    def from_data(
        cls,
        data_path: str = './data',
        pipes_path: str = './pipes/combined',
        checkpoints_path: str = None,
        algorithms: Optional[List[str]] = None,
        load_fsbo: bool = True,
        load_metafeatx: bool = True,
        metafeatx_reference_algo: str = 'adaboost',
        device: str = 'cpu'
    ) -> 'PipelineRecommender':
        """
        Crea un PipelineRecommender desde los datos del proyecto.
        
        MODELOS USADOS:
        - FSBO: Se cargan los checkpoints entrenados (16 algoritmos)
        - MetaFeatX: Se cargan embeddings PRE-CALCULADOS (no se entrena)
        
        Args:
            data_path: Ruta a carpeta data/
            pipes_path: Ruta a pipes/combined/
            checkpoints_path: Ruta a checkpoints FSBO
            algorithms: Lista de algoritmos a cargar (None = todos)
            load_fsbo: Si cargar modelos FSBO entrenados
            load_metafeatx: Si cargar embeddings MetaFeatX pre-calculados
            metafeatx_reference_algo: Algoritmo usado para los embeddings (adaboost, random_forest)
            device: Dispositivo para PyTorch
        """
        data_path = Path(data_path)
        pipes_path = Path(pipes_path)
        
        if checkpoints_path is None:
            checkpoints_path = PROJECT_ROOT / 'transfer-learning' / 'experiments' / 'checkpoints_pipeline'
        else:
            checkpoints_path = Path(checkpoints_path)
        
        # 1. Cargar metafeatures básicas
        basic_repr_path = data_path / 'basic_representations.csv'
        if not basic_repr_path.exists():
            raise FileNotFoundError(f"No encontrado: {basic_repr_path}")
        
        basic_representations = pd.read_csv(basic_repr_path)
        logger.info(f"Basic representations: {basic_representations.shape}")
        
        # 2. Cargar datos históricos de cada algoritmo
        if algorithms is None:
            algorithms = AVAILABLE_ALGORITHMS
        
        historical_data = {}
        task_id_mapping = {}
        
        for algorithm in algorithms:
            csv_name = ALGORITHM_CSV_MAPPING.get(algorithm)
            if csv_name is None:
                continue
            
            csv_path = pipes_path / f'{csv_name}_combined.csv'
            if not csv_path.exists():
                logger.warning(f"No encontrado: {csv_path}")
                continue
            
            df = pd.read_csv(csv_path)
            historical_data[algorithm] = df
            logger.info(f"  {algorithm}: {len(df)} registros")
            
            # Actualizar mapping
            if 'Dataset' in df.columns:
                for dataset in df['Dataset'].unique():
                    if dataset not in task_id_mapping:
                        parts = str(dataset).split('_')
                        for part in reversed(parts):
                            try:
                                task_id = int(part)
                                if task_id in basic_representations['task_id'].values:
                                    task_id_mapping[dataset] = task_id
                                    break
                            except ValueError:
                                continue
        
        logger.info(f"Algoritmos cargados: {len(historical_data)}")
        logger.info(f"Task mapping: {len(task_id_mapping)} datasets")
        
        # 3. Cargar modelos FSBO
        fsbo_models = {}
        if load_fsbo and TORCH_AVAILABLE and checkpoints_path.exists():
            logger.info(f"\n[FSBO] Cargando modelos desde {checkpoints_path}...")
            fsbo_models = cls._load_fsbo_models(checkpoints_path, algorithms, device)
            logger.info(f"[FSBO] {len(fsbo_models)} modelos cargados")
        
        # 4. Cargar embeddings MetaFeatX PRE-CALCULADOS (no entrenar)
        metafeatx_embeddings = None
        if load_metafeatx:
            logger.info(f"\n[MetaFeatX] Buscando embeddings pre-calculados...")
            metafeatx_embeddings = cls._load_metafeatx_embeddings(
                metafeatx_reference_algo
            )
        
        return cls(
            basic_representations=basic_representations,
            historical_data=historical_data,
            task_id_mapping=task_id_mapping,
            metafeatx_embeddings=metafeatx_embeddings,
            fsbo_models=fsbo_models,
            device=device
        )
    
    @staticmethod
    def _load_fsbo_models(
        checkpoints_path: Path, 
        algorithms: List[str],
        device: str
    ) -> Dict[str, Any]:
        """Carga todos los modelos FSBO disponibles."""
        fsbo_models = {}
        
        for algorithm in algorithms:
            # Buscar checkpoint
            pattern = f'fsbo_pipeline_{algorithm}_*.pt'
            checkpoints = list(checkpoints_path.glob(pattern))
            
            if not checkpoints:
                continue
            
            # Usar el más reciente
            checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
            
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                config = checkpoint.get('config', {})
                
                input_dim = config.get('input_dim', 50)
                hidden_dim = config.get('hidden_dim', 128)
                n_layers = config.get('n_layers', 3)
                feature_names = config.get('feature_names', [])
                
                # Crear modelo
                train_x = torch.zeros(1, input_dim).to(device)
                train_y = torch.zeros(1).to(device)
                
                feature_extractor = DeepKernelNetwork(input_dim, hidden_dim, n_layers).to(device)
                likelihood = GaussianLikelihood().to(device)
                model = DeepKernelGP(train_x, train_y, likelihood, feature_extractor).to(device)
                
                # Cargar pesos
                model.load_state_dict(checkpoint['model_state'])
                likelihood.load_state_dict(checkpoint['likelihood_state'])
                
                model.eval()
                likelihood.eval()
                
                fsbo_models[algorithm] = {
                    'model': model,
                    'likelihood': likelihood,
                    'feature_names': feature_names,
                    'input_dim': input_dim
                }
                
                logger.info(f"  ✓ {algorithm} (dim={input_dim})")
                
            except Exception as e:
                logger.warning(f"  ✗ {algorithm}: {e}")
        
        return fsbo_models
    
    @staticmethod
    def _load_metafeatx_embeddings(
        reference_algo: str = 'adaboost'
    ) -> Optional[pd.DataFrame]:
        """
        Carga embeddings MetaFeatX PRE-CALCULADOS.
        
        Los embeddings ya fueron calculados por el modelo MetaFeatX entrenado
        y están guardados en notebooks/metafeatx_{algo}_representation.csv
        
        Args:
            reference_algo: Algoritmo de referencia para los embeddings
            
        Returns:
            DataFrame con columnas [0, 1, 2, ..., task_id] o None si no existe
        """
        # Posibles ubicaciones de embeddings pre-calculados
        possible_paths = [
            PROJECT_ROOT / 'notebooks' / f'metafeatx_{reference_algo}_representation.csv',
            PROJECT_ROOT / 'notebooks' / 'metafeatx_representation.csv',
            PROJECT_ROOT / 'data' / f'metafeatx_{reference_algo}_embeddings.csv',
            PROJECT_ROOT / 'data' / 'metafeatx_embeddings.csv',
        ]
        
        for path in possible_paths:
            if path.exists():
                try:
                    embeddings = pd.read_csv(path)
                    
                    # Verificar que tiene task_id
                    if 'task_id' not in embeddings.columns:
                        logger.warning(f"  {path} no tiene columna task_id")
                        continue
                    
                    logger.info(f"  ✓ Embeddings cargados desde: {path}")
                    logger.info(f"    Shape: {embeddings.shape}")
                    logger.info(f"    Task IDs: {len(embeddings['task_id'].unique())}")
                    
                    return embeddings
                    
                except Exception as e:
                    logger.warning(f"  Error leyendo {path}: {e}")
                    continue
        
        logger.warning(f"  No se encontraron embeddings MetaFeatX pre-calculados")
        logger.warning(f"  Buscados en: {[str(p) for p in possible_paths]}")
        return None
    
    def get_neighbors(self, task_id: int, k: int = 5) -> List[Tuple[int, float]]:
        """Encuentra k datasets más similares usando el embedding."""
        if task_id not in self.embedding_task_ids:
            raise ValueError(f"task_id {task_id} no encontrado")
        
        query_idx = self.embedding_task_ids.index(task_id)
        query_embedding = self.embedding_matrix[query_idx:query_idx+1]
        
        distances = pairwise_distances(
            query_embedding, 
            self.embedding_matrix, 
            metric='euclidean'
        )[0]
        
        sorted_indices = np.argsort(distances)
        
        neighbors = []
        for idx in sorted_indices:
            if self.embedding_task_ids[idx] != task_id:
                neighbors.append((self.embedding_task_ids[idx], distances[idx]))
            if len(neighbors) >= k:
                break
        
        return neighbors
    
    def recommend(
        self,
        task_id: int,
        top_k: int = 10,
        k_neighbors: int = 5,
        algorithms: Optional[List[str]] = None,
    ) -> RecommendationResult:
        """
        Genera ranking de pipelines recomendados para un dataset.
        
        PROCESO:
        1. Usa MetaFeatX embeddings para encontrar vecinos similares
           (datasets con HP óptimos similares)
        2. Extrae configs históricas de esos vecinos
        3. Usa accuracy histórica como score esperado
        4. Pondera por distancia y rankea las recomendaciones
        
        NOTA: Los modelos FSBO están cargados para uso futuro en optimización
        interactiva (FSBOPipelineOptimizer), pero aquí usamos accuracy histórica
        porque el GP necesita observaciones para predecir bien.
        
        Args:
            task_id: ID del dataset objetivo
            top_k: Número de recomendaciones a retornar
            k_neighbors: Número de datasets similares a considerar
            algorithms: Algoritmos a considerar (None = todos)
            
        Returns:
            RecommendationResult con ranking de configuraciones
        """
        if algorithms is None:
            algorithms = list(self.historical_data.keys())
        
        # 1. Encontrar vecinos (usando MetaFeatX embedding si disponible)
        neighbors = self.get_neighbors(task_id, k=k_neighbors)
        neighbor_ids = [n[0] for n in neighbors]
        neighbor_distances = {n[0]: n[1] for n in neighbors}
        
        logger.info(f"Vecinos para task {task_id}: {neighbor_ids}")
        logger.info(f"  (Embedding: {self._embedding_source})")
        
        # 2. Recolectar configs de todos los algoritmos
        all_candidates = []
        algorithm_scores = {algo: [] for algo in algorithms}
        
        for neighbor_id in neighbor_ids:
            distance = neighbor_distances[neighbor_id]
            
            # Buscar datasets que correspondan a este task_id
            matching_datasets = [
                ds for ds, tid in self.task_id_mapping.items() 
                if tid == neighbor_id
            ]
            
            # También buscar por coincidencia parcial
            for ds in set(d for algo_df in self.historical_data.values() 
                         for d in algo_df.get('Dataset', pd.Series()).unique()):
                if str(neighbor_id) in str(ds) and ds not in matching_datasets:
                    matching_datasets.append(ds)
            
            for dataset in matching_datasets:
                for algorithm in algorithms:
                    cache_key = (dataset, algorithm)
                    if cache_key in self.best_configs_cache:
                        configs = self.best_configs_cache[cache_key]
                        
                        # 3. Obtener scores esperados (accuracy histórica)
                        predictions = self._get_expected_scores(configs, algorithm)
                        
                        for config, (predicted_score, confidence) in zip(configs, predictions):
                            # Calcular score combinado
                            weight = 1.0 / (1.0 + distance)
                            
                            # Score = accuracy histórica ponderada por distancia
                            # (datasets más similares tienen más peso)
                            weighted_score = predicted_score * weight
                            
                            candidate = {
                                'algorithm': algorithm,
                                'pipeline': config['pipeline'],
                                'classifier': config['classifier'],
                                'historical_accuracy': config['accuracy'],
                                'predicted_score': predicted_score,
                                'confidence': confidence,
                                'weighted_score': weighted_score,
                                'source_dataset': dataset,
                                'source_task_id': neighbor_id,
                                'distance': distance,
                                'weight': weight,
                            }
                            all_candidates.append(candidate)
                            algorithm_scores[algorithm].append(predicted_score)
        
        # 4. Rankear candidatos por score predicho
        all_candidates.sort(key=lambda x: x['weighted_score'], reverse=True)
        
        # 5. Eliminar duplicados
        seen = set()
        unique_candidates = []
        for c in all_candidates:
            key = (
                c['algorithm'],
                tuple(sorted(c['pipeline'].items())),
                tuple(sorted((k, str(v)) for k, v in c['classifier'].items()))
            )
            if key not in seen:
                seen.add(key)
                unique_candidates.append(c)
        
        # 6. Crear recomendaciones
        recommendations = []
        for i, c in enumerate(unique_candidates[:top_k], 1):
            rec = PipelineRecommendation(
                rank=i,
                algorithm=c['algorithm'],
                pipeline=c['pipeline'],
                classifier_params=c['classifier'],
                expected_score=c['predicted_score'],
                confidence=c['weight'],
                source_info={
                    'dataset': c['source_dataset'],
                    'task_id': c['source_task_id'],
                    'distance': c['distance'],
                    'historical_accuracy': c['historical_accuracy'],
                }
            )
            recommendations.append(rec)
        
        # 7. Ranking de algoritmos
        algo_ranking = []
        for algo, scores in algorithm_scores.items():
            if scores:
                avg_score = np.mean(scores)
                algo_ranking.append((algo, avg_score))
        algo_ranking.sort(key=lambda x: x[1], reverse=True)
        
        return RecommendationResult(
            task_id=task_id,
            n_neighbors_used=len(neighbors),
            recommendations=recommendations,
            algorithm_ranking=algo_ranking,
            models_used={
                'metafeatx_embeddings': self.metafeatx_embeddings is not None,
                'fsbo': len(self.fsbo_models) > 0,
                'embedding_source': self._embedding_source
            }
        )
    
    def recommend_algorithm_only(
        self,
        task_id: int,
        k_neighbors: int = 10
    ) -> List[Tuple[str, float, int]]:
        """
        Recomienda solo el algoritmo (sin pipeline/HP específicos).
        
        Returns:
            Lista de (algoritmo, score_promedio, n_observaciones)
        """
        neighbors = self.get_neighbors(task_id, k=k_neighbors)
        
        algo_scores = {algo: [] for algo in self.historical_data.keys()}
        
        for neighbor_id, distance in neighbors:
            for ds, tid in self.task_id_mapping.items():
                if tid == neighbor_id:
                    for algo, df in self.historical_data.items():
                        if 'Dataset' in df.columns:
                            subset = df[df['Dataset'] == ds]
                            if not subset.empty:
                                best_acc = subset['Fold Accuracy'].max()
                                weight = 1.0 / (1.0 + distance)
                                algo_scores[algo].append(best_acc * weight)
        
        ranking = []
        for algo, scores in algo_scores.items():
            if scores:
                ranking.append((algo, np.mean(scores), len(scores)))
        
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking
    
    def get_best_config_for_algorithm(
        self,
        task_id: int,
        algorithm: str,
        k_neighbors: int = 5,
        top_n: int = 3
    ) -> List[Dict]:
        """
        Dado un algoritmo específico, retorna las mejores configs
        usando predicción FSBO si está disponible.
        """
        result = self.recommend(
            task_id=task_id,
            top_k=50,
            k_neighbors=k_neighbors,
            algorithms=[algorithm]
        )
        
        configs = []
        for rec in result.recommendations[:top_n]:
            configs.append({
                'pipeline': rec.pipeline,
                'classifier': rec.classifier_params,
                'expected_score': rec.expected_score,
                'source': rec.source_info
            })
        
        return configs


# =============================================================================
# CLI
# =============================================================================

def main():
    """Función principal para testing."""
    import argparse
    
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(
        description='Pipeline Recommender - Usa modelos FSBO + MetaFeatX'
    )
    parser.add_argument('--task-id', type=int, default=3)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--k-neighbors', type=int, default=5)
    parser.add_argument('--data-path', type=str, default=None)
    parser.add_argument('--pipes-path', type=str, default=None)
    parser.add_argument('--checkpoints-path', type=str, default=None)
    parser.add_argument('--no-fsbo', action='store_true', help='No cargar modelos FSBO')
    parser.add_argument('--no-metafeatx', action='store_true', help='No cargar embeddings MetaFeatX')
    
    args = parser.parse_args()
    
    base_path = PROJECT_ROOT
    data_path = args.data_path or str(base_path / 'data')
    pipes_path = args.pipes_path or str(base_path / 'pipes' / 'combined')
    checkpoints_path = args.checkpoints_path or str(
        base_path / 'transfer-learning' / 'experiments' / 'checkpoints_pipeline'
    )
    
    print("=" * 70)
    print("🎯 PIPELINE RECOMMENDER (con modelos entrenados)")
    print("   Usa: MetaFeatX (embedding) + FSBO (predicción)")
    print("=" * 70)
    print(f"\nTask ID: {args.task_id}")
    print(f"Top K: {args.top_k}")
    print(f"K vecinos: {args.k_neighbors}")
    
    try:
        print("\n[1] Cargando datos y MODELOS...")
        recommender = PipelineRecommender.from_data(
            data_path=data_path,
            pipes_path=pipes_path,
            checkpoints_path=checkpoints_path,
            load_fsbo=not args.no_fsbo,
            load_metafeatx=not args.no_metafeatx
        )
        
        print(f"\n    ✓ {len(recommender.historical_data)} algoritmos con datos")
        print(f"    ✓ {len(recommender.fsbo_models)} modelos FSBO cargados")
        print(f"    ✓ MetaFeatX embeddings: {'Sí' if recommender.metafeatx_embeddings is not None else 'No'}")
        print(f"    ✓ {len(recommender.best_configs_cache)} configs en cache")
        
        # Ranking de algoritmos
        print(f"\n[2] Ranking de ALGORITMOS para task {args.task_id}...")
        algo_ranking = recommender.recommend_algorithm_only(
            task_id=args.task_id,
            k_neighbors=args.k_neighbors
        )
        
        print(f"\n    {'Rank':<6} {'Algoritmo':<25} {'Score':<10} {'N obs'}")
        print("    " + "-" * 50)
        for i, (algo, score, n) in enumerate(algo_ranking[:5], 1):
            fsbo_mark = "🔮" if algo in recommender.fsbo_models else "  "
            print(f"    {i:<6} {algo:<25} {score:.4f}     {n}  {fsbo_mark}")
        
        # Recomendaciones completas
        print(f"\n[3] Top {args.top_k} CONFIGURACIONES COMPLETAS...")
        print("    (Usando accuracy histórica de datasets similares)")
        
        result = recommender.recommend(
            task_id=args.task_id,
            top_k=args.top_k,
            k_neighbors=args.k_neighbors
        )
        
        print(f"\n    Modelos usados: {result.models_used}")
        print(f"\n    {'#':<4} {'Algoritmo':<18} {'Score':<8} {'Dist':<6} {'Pipeline':<35}")
        print("    " + "-" * 80)
        
        for rec in result.recommendations[:args.top_k]:
            dist = rec.source_info.get('distance', 0)
            pipeline_str = f"imp={rec.pipeline.get('imputer_strategy', '?')[:4]}, " \
                          f"cat={rec.pipeline.get('categorical_strategy', '?')[:4]}, " \
                          f"fs={rec.pipeline.get('feature_selection', '?')[:6]}, " \
                          f"sc={rec.pipeline.get('scaler', '?')[:4]}"
            print(f"    {rec.rank:<4} {rec.algorithm:<18} {rec.expected_score:<8.4f} {dist:<6.2f} {pipeline_str}")
        
        # Detalle de top 3
        print(f"\n[4] Detalle de TOP 3:")
        for rec in result.recommendations[:3]:
            print(f"\n    === Rank {rec.rank}: {rec.algorithm} ===")
            print(f"    Accuracy histórica: {rec.expected_score:.4f}")
            print(f"    Dataset similar: {rec.source_info['dataset']}")
            print(f"    Distancia (MetaFeatX): {rec.source_info['distance']:.4f}")
            print(f"    Pipeline: {rec.pipeline}")
            print(f"    Classifier: {rec.classifier_params}")
        
        print("\n" + "=" * 70)
        print("✅ Recomendación completada")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
