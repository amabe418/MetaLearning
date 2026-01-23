"""
FSBOPipelineOptimizer: Optimizador de pipeline completo + hiperparámetros.

Esta clase proporciona una API limpia (observe/suggest) para optimizar
no solo hiperparámetros del clasificador, sino todo el pipeline de ML:
- Estrategia de imputación
- Codificación de categóricos
- Feature selection
- Scaling
- Hiperparámetros del clasificador

Uso básico:
    optimizer = FSBOPipelineOptimizer.from_pretrained('adaboost')
    
    # Loop de optimización
    for _ in range(budget):
        config = optimizer.suggest()
        # config = {
        #     'pipeline': {'imputer': 'simpleimputer', 'scaler': 'standard', ...},
        #     'classifier': {'n_estimators': 100, 'learning_rate': 0.1, ...}
        # }
        score = evaluate_pipeline(config)
        optimizer.observe(config, score)
    
    best_config = optimizer.get_best()

Autor: Proyecto académico MetaLearning
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from scipy.stats import norm

logger = logging.getLogger(__name__)


# =============================================================================
# Modelos
# =============================================================================

class DeepKernelNetwork(nn.Module):
    """Red neuronal para pipeline completo."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, n_layers: int = 3):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.1))
        
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        
        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dim
        
    def forward(self, x):
        return self.network(x)


class DeepKernelGP(ExactGP):
    """Gaussian Process con Deep Kernel."""
    
    def __init__(self, train_x, train_y, likelihood, feature_extractor):
        super().__init__(train_x, train_y, likelihood)
        self.feature_extractor = feature_extractor
        self.mean_module = ConstantMean()
        latent_dim = feature_extractor.output_dim
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))
        
    def forward(self, x):
        projected_x = self.feature_extractor(x)
        mean = self.mean_module(projected_x)
        covar = self.covar_module(projected_x)
        return MultivariateNormal(mean, covar)


# =============================================================================
# Estructuras de datos
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuración completa de un pipeline."""
    imputer_strategy: str
    categorical_strategy: str
    feature_selection: str
    scaler: str
    classifier_params: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return {
            'pipeline': {
                'imputer_strategy': self.imputer_strategy,
                'categorical_strategy': self.categorical_strategy,
                'feature_selection': self.feature_selection,
                'scaler': self.scaler,
            },
            'classifier': self.classifier_params.copy()
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'PipelineConfig':
        return cls(
            imputer_strategy=d['pipeline']['imputer_strategy'],
            categorical_strategy=d['pipeline']['categorical_strategy'],
            feature_selection=d['pipeline']['feature_selection'],
            scaler=d['pipeline']['scaler'],
            classifier_params=d['classifier'].copy()
        )


@dataclass
class PipelineOptimizationResult:
    """Resultado de optimización de pipeline."""
    algorithm: str
    best_config: PipelineConfig
    best_score: float
    n_evaluations: int
    history: List[float]
    all_configs: List[PipelineConfig]
    all_scores: List[float]


# =============================================================================
# Espacio de Pipeline
# =============================================================================

PIPELINE_OPTIONS = {
    'imputer_strategy': ['none', 'simpleimputer'],
    'categorical_strategy': ['none', 'onehot', 'ordinalencoder'],
    'feature_selection': [
        'extra_tree', 'fastica', 'feature_agglomeration', 'generic_univariate',
        'kernel_pca', 'linear_svc', 'none', 'nystroem', 'pca',
        'polynomial_features', 'random_trees_embedding', 'rbf_sampler',
        'select_percentile', 'truncated_svd'
    ],
    'scaler': ['minmax', 'none', 'normalizer', 'power', 'quantile', 'robust', 'standard'],
}

# Definición de hiperparámetros por algoritmo (simplificado)
CLASSIFIER_PARAMS = {
    'adaboost': {
        'n_estimators': {'type': 'int', 'range': (10, 500)},
        'learning_rate': {'type': 'float', 'range': (0.01, 10.0), 'log': True},
        'base_estimator__max_depth': {'type': 'int', 'range': (1, 10)},
    },
    'random_forest': {
        'n_estimators': {'type': 'int', 'range': (10, 500)},
        'max_depth': {'type': 'int', 'range': (1, 50)},
        'min_samples_split': {'type': 'int', 'range': (2, 20)},
        'min_samples_leaf': {'type': 'int', 'range': (1, 20)},
        'criterion': {'type': 'categorical', 'choices': ['gini', 'entropy', 'log_loss']},
        'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', '']},
        'bootstrap': {'type': 'categorical', 'choices': ['True', 'False']},
    },
    'svc': {
        'C': {'type': 'float', 'range': (0.001, 100.0), 'log': True},
        'kernel': {'type': 'categorical', 'choices': ['linear', 'poly', 'rbf', 'sigmoid']},
        'degree': {'type': 'int', 'range': (1, 5)},
        'gamma': {'type': 'categorical', 'choices': ['scale', 'auto']},
        'coef0': {'type': 'float', 'range': (-1.0, 1.0)},
        'tol': {'type': 'float', 'range': (0.0001, 0.01), 'log': True},
        'shrinking': {'type': 'categorical', 'choices': ['True', 'False']},
    },
    'mlp': {
        'alpha': {'type': 'float', 'range': (0.0001, 1.0), 'log': True},
        'learning_rate_init': {'type': 'float', 'range': (0.001, 1.0), 'log': True},
        'max_iter': {'type': 'int', 'range': (100, 1000)},
        'activation': {'type': 'categorical', 'choices': ['identity', 'logistic', 'tanh', 'relu']},
        'solver': {'type': 'categorical', 'choices': ['lbfgs', 'sgd', 'adam']},
        'learning_rate': {'type': 'categorical', 'choices': ['constant', 'invscaling', 'adaptive']},
    },
    'lda': {
        'tol': {'type': 'float', 'range': (0.0001, 0.01), 'log': True},
        'solver': {'type': 'categorical', 'choices': ['svd', 'lsqr', 'eigen']},
        'shrinkage': {'type': 'categorical', 'choices': ['auto', 'None']},
    },
    'qda': {
        'reg_param': {'type': 'float', 'range': (0.0, 1.0)},
        'tol': {'type': 'float', 'range': (0.0001, 0.01), 'log': True},
    },
    'gaussian_nb': {
        'var_smoothing': {'type': 'float', 'range': (1e-12, 1e-6), 'log': True},
    },
    'bernoulli_nb': {
        'alpha': {'type': 'float', 'range': (0.0, 10.0)},
        'binarize': {'type': 'float', 'range': (0.0, 1.0)},
    },
    'multinomial_nb': {
        'alpha': {'type': 'float', 'range': (0.0, 10.0)},
    },
    'decision_tree': {
        'max_depth': {'type': 'int', 'range': (1, 50)},
        'min_samples_split': {'type': 'int', 'range': (2, 20)},
        'min_samples_leaf': {'type': 'int', 'range': (1, 20)},
        'criterion': {'type': 'categorical', 'choices': ['gini', 'entropy']},
        'splitter': {'type': 'categorical', 'choices': ['best', 'random']},
        'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', '']},
    },
    'extra_trees': {
        'n_estimators': {'type': 'int', 'range': (10, 500)},
        'max_depth': {'type': 'int', 'range': (1, 50)},
        'min_samples_split': {'type': 'int', 'range': (2, 20)},
        'min_samples_leaf': {'type': 'int', 'range': (1, 20)},
        'criterion': {'type': 'categorical', 'choices': ['gini', 'entropy']},
        'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', '']},
        'bootstrap': {'type': 'categorical', 'choices': ['True', 'False']},
    },
    'kneighbors': {
        'n_neighbors': {'type': 'int', 'range': (1, 50)},
        'leaf_size': {'type': 'int', 'range': (10, 100)},
        'p': {'type': 'int', 'range': (1, 5)},
        'weights': {'type': 'categorical', 'choices': ['uniform', 'distance']},
        'algorithm': {'type': 'categorical', 'choices': ['auto', 'ball_tree', 'kd_tree', 'brute']},
        'metric': {'type': 'categorical', 'choices': ['euclidean', 'manhattan', 'minkowski']},
    },
    'linear_svc': {
        'C': {'type': 'float', 'range': (0.001, 100.0), 'log': True},
        'tol': {'type': 'float', 'range': (0.0001, 0.01), 'log': True},
        'max_iter': {'type': 'int', 'range': (100, 10000)},
        'penalty': {'type': 'categorical', 'choices': ['l1', 'l2']},
        'loss': {'type': 'categorical', 'choices': ['hinge', 'squared_hinge']},
        'dual': {'type': 'categorical', 'choices': ['True', 'False']},
    },
    'sgd': {
        'alpha': {'type': 'float', 'range': (0.0001, 1.0), 'log': True},
        'l1_ratio': {'type': 'float', 'range': (0.0, 1.0)},
        'max_iter': {'type': 'int', 'range': (100, 10000)},
        'tol': {'type': 'float', 'range': (0.0001, 0.01), 'log': True},
        'loss': {'type': 'categorical', 'choices': ['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron']},
        'penalty': {'type': 'categorical', 'choices': ['l1', 'l2', 'elasticnet']},
        'learning_rate': {'type': 'categorical', 'choices': ['constant', 'optimal', 'invscaling', 'adaptive']},
    },
    'hist_gradient_boosting': {
        'learning_rate': {'type': 'float', 'range': (0.01, 1.0), 'log': True},
        'max_iter': {'type': 'int', 'range': (50, 500)},
        'max_leaf_nodes': {'type': 'int', 'range': (10, 255)},
        'max_depth': {'type': 'int', 'range': (1, 50)},
        'min_samples_leaf': {'type': 'int', 'range': (1, 100)},
        'l2_regularization': {'type': 'float', 'range': (0.0, 10.0)},
    },
    'passive_aggressive': {
        'C': {'type': 'float', 'range': (0.001, 10.0), 'log': True},
        'tol': {'type': 'float', 'range': (0.0001, 0.01), 'log': True},
        'max_iter': {'type': 'int', 'range': (100, 10000)},
        'loss': {'type': 'categorical', 'choices': ['hinge', 'squared_hinge']},
    },
}


class PipelineSpace:
    """
    Define y maneja el espacio de búsqueda de pipeline completo.
    
    Usa los feature_names del checkpoint para codificar/decodificar
    configuraciones de manera consistente con el modelo entrenado.
    """
    
    def __init__(self, algorithm: str, feature_names: List[str]):
        self.algorithm = algorithm
        self.feature_names = feature_names
        self.classifier_params = CLASSIFIER_PARAMS.get(algorithm, {})
        
        # Parsear feature_names para entender la estructura
        self._parse_feature_names()
    
    def _parse_feature_names(self):
        """Parsea los feature_names para entender la estructura de encoding."""
        self.pipeline_groups = {}  # nombre -> [(idx, opcion), ...]
        self.hp_numeric = {}       # nombre -> idx
        self.hp_categorical = {}   # nombre -> [(idx, opcion), ...]
        
        # Primero identificar qué hiperparámetros son categóricos
        categorical_params = {}
        for param_name, param_spec in self.classifier_params.items():
            if param_spec.get('type') == 'categorical':
                categorical_params[param_name] = param_spec.get('choices', [])
        
        for idx, name in enumerate(self.feature_names):
            if name.startswith('Imputer Strategy_'):
                option = name.split('_', 2)[-1]
                self.pipeline_groups.setdefault('imputer_strategy', []).append((idx, option))
            elif name.startswith('Categorical Strategy_'):
                option = name.split('_', 2)[-1]
                self.pipeline_groups.setdefault('categorical_strategy', []).append((idx, option))
            elif name.startswith('Feature Selection_'):
                option = name.split('_', 2)[-1]
                self.pipeline_groups.setdefault('feature_selection', []).append((idx, option))
            elif name.startswith('Scaler_'):
                option = name.split('_', 1)[-1]
                self.pipeline_groups.setdefault('scaler', []).append((idx, option))
            elif name.startswith('hp_'):
                # Hiperparámetro del clasificador
                hp_part = name[3:]  # Quitar "hp_"
                
                # Buscar si coincide con algún parámetro categórico
                matched_categorical = False
                for param_name, choices in categorical_params.items():
                    prefix = f"{param_name}_"
                    if hp_part.startswith(prefix) or hp_part == param_name:
                        # Extraer el valor de la opción
                        if hp_part.startswith(prefix):
                            choice = hp_part[len(prefix):]
                        else:
                            choice = ""
                        self.hp_categorical.setdefault(param_name, []).append((idx, choice))
                        matched_categorical = True
                        break
                
                if not matched_categorical:
                    # Es numérico
                    self.hp_numeric[hp_part] = idx
    
    def get_dim(self) -> int:
        """Retorna dimensión total del espacio."""
        return len(self.feature_names)
    
    def sample_random(self, n: int = 1) -> np.ndarray:
        """Genera n configuraciones aleatorias válidas."""
        dim = self.get_dim()
        X = np.zeros((n, dim), dtype=np.float32)
        
        for i in range(n):
            # Pipeline: seleccionar uno aleatorio de cada grupo
            for group_name, options in self.pipeline_groups.items():
                chosen_idx = np.random.randint(len(options))
                for j, (idx, _) in enumerate(options):
                    X[i, idx] = 1.0 if j == chosen_idx else 0.0
            
            # HP categóricos: seleccionar uno aleatorio
            for hp_name, options in self.hp_categorical.items():
                chosen_idx = np.random.randint(len(options))
                for j, (idx, _) in enumerate(options):
                    X[i, idx] = 1.0 if j == chosen_idx else 0.0
            
            # HP numéricos: valor aleatorio [0, 1]
            for hp_name, idx in self.hp_numeric.items():
                X[i, idx] = np.random.rand()
        
        return X
    
    def decode(self, x: np.ndarray) -> PipelineConfig:
        """Decodifica vector a configuración de pipeline."""
        # Pipeline options
        pipeline_choices = {}
        for group_name, options in self.pipeline_groups.items():
            probs = [x[idx] for idx, _ in options]
            best_idx = np.argmax(probs)
            pipeline_choices[group_name] = options[best_idx][1]
        
        # Classifier params
        classifier_params = {}
        
        # HP categóricos
        for hp_name, options in self.hp_categorical.items():
            probs = [x[idx] for idx, _ in options]
            best_idx = np.argmax(probs)
            value = options[best_idx][1]
            # Convertir strings booleanos
            if value == 'True':
                value = True
            elif value == 'False':
                value = False
            # Valor vacío -> None (para parámetros como max_features)
            if value == '':
                value = None
            classifier_params[hp_name] = value
        
        # HP numéricos
        for hp_name, idx in self.hp_numeric.items():
            val = x[idx]
            param_spec = self.classifier_params.get(hp_name, {})
            low, high = param_spec.get('range', (0, 1))
            
            if param_spec.get('log', False) and low > 0:
                value = np.exp(np.log(low) + val * (np.log(high) - np.log(low)))
            else:
                value = low + val * (high - low)
            
            if param_spec.get('type') == 'int':
                value = int(round(value))
            
            classifier_params[hp_name] = value
        
        return PipelineConfig(
            imputer_strategy=pipeline_choices.get('imputer_strategy', 'simpleimputer'),
            categorical_strategy=pipeline_choices.get('categorical_strategy', 'onehot'),
            feature_selection=pipeline_choices.get('feature_selection', 'none'),
            scaler=pipeline_choices.get('scaler', 'standard'),
            classifier_params=classifier_params
        )
    
    def encode(self, config: PipelineConfig) -> np.ndarray:
        """Codifica configuración a vector."""
        x = np.zeros(len(self.feature_names), dtype=np.float32)
        
        # Pipeline options
        pipeline_values = {
            'imputer_strategy': config.imputer_strategy,
            'categorical_strategy': config.categorical_strategy,
            'feature_selection': config.feature_selection,
            'scaler': config.scaler,
        }
        
        for group_name, options in self.pipeline_groups.items():
            current_value = pipeline_values.get(group_name, '')
            for idx, option in options:
                x[idx] = 1.0 if option == current_value else 0.0
        
        # HP categóricos
        for hp_name, options in self.hp_categorical.items():
            value = config.classifier_params.get(hp_name)
            val_str = str(value) if value is not None else ''
            for idx, option in options:
                x[idx] = 1.0 if option == val_str else 0.0
        
        # HP numéricos
        for hp_name, idx in self.hp_numeric.items():
            value = config.classifier_params.get(hp_name)
            param_spec = self.classifier_params.get(hp_name, {})
            
            if value is None:
                x[idx] = 0.5
            else:
                low, high = param_spec.get('range', (0, 1))
                if param_spec.get('log', False) and low > 0 and value > 0:
                    x[idx] = (np.log(value) - np.log(low)) / (np.log(high) - np.log(low))
                else:
                    x[idx] = (value - low) / (high - low) if high > low else 0.5
                x[idx] = np.clip(x[idx], 0, 1)
        
        return x


# =============================================================================
# FSBOPipelineOptimizer
# =============================================================================

class FSBOPipelineOptimizer:
    """
    Optimizador de pipeline completo basado en FSBO.
    
    Sugiere tanto configuración del pipeline (preprocesamiento) como
    hiperparámetros del clasificador.
    """
    
    def __init__(
        self,
        algorithm: str,
        model: DeepKernelGP,
        likelihood: GaussianLikelihood,
        pipeline_space: PipelineSpace,
        feature_names: List[str],
        device: str = 'cpu'
    ):
        self.algorithm = algorithm
        self.model = model
        self.likelihood = likelihood
        self.pipeline_space = pipeline_space
        self.feature_names = feature_names
        self.device = device
        
        # Observaciones
        self.X_observed: List[np.ndarray] = []
        self.y_observed: List[float] = []
        self.configs_observed: List[PipelineConfig] = []
        
        # Historial
        self.best_y_history: List[float] = []
        
        # Configuración
        self.xi = 0.01
        self.finetune_frequency = 5
        self.finetune_epochs = 20
        self._n_suggests = 0
    
    @classmethod
    def from_pretrained(
        cls,
        algorithm: str,
        checkpoint_dir: Optional[str] = None,
        device: str = 'cpu'
    ) -> 'FSBOPipelineOptimizer':
        """
        Carga un optimizador desde checkpoint pre-entrenado.
        
        Args:
            algorithm: Nombre del algoritmo
            checkpoint_dir: Directorio de checkpoints
            device: Dispositivo
            
        Returns:
            FSBOPipelineOptimizer configurado
        """
        if checkpoint_dir is None:
            checkpoint_dir = Path(__file__).parent.parent / 'experiments' / 'checkpoints_pipeline'
        else:
            checkpoint_dir = Path(checkpoint_dir)
        
        # Buscar checkpoint
        checkpoints = list(checkpoint_dir.glob(f'fsbo_pipeline_{algorithm}_*.pt'))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint found for {algorithm} in {checkpoint_dir}")
        
        checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
        
        # Cargar checkpoint
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
        
        # Crear espacio de pipeline
        pipeline_space = PipelineSpace(algorithm, feature_names)
        
        logger.info(f"Loaded FSBO Pipeline optimizer for {algorithm}")
        logger.info(f"  Input dim: {input_dim}")
        logger.info(f"  Features: {len(feature_names)}")
        
        return cls(
            algorithm=algorithm,
            model=model,
            likelihood=likelihood,
            pipeline_space=pipeline_space,
            feature_names=feature_names,
            device=device
        )
    
    def suggest(self, n_candidates: int = 1000) -> Dict[str, Any]:
        """
        Sugiere la siguiente configuración de pipeline a evaluar.
        
        Returns:
            Diccionario con configuración de pipeline y clasificador
        """
        self._n_suggests += 1
        
        # Generar candidatos
        X_candidates = self.pipeline_space.sample_random(n_candidates)
        X_candidates_tensor = torch.tensor(X_candidates, dtype=torch.float32).to(self.device)
        
        # Predecir
        self.model.eval()
        self.likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.model(X_candidates_tensor))
            mu = pred.mean.cpu().numpy()
            sigma = pred.stddev.cpu().numpy()
        
        # Expected Improvement
        y_best = max(self.y_observed) if self.y_observed else 0.0
        ei = self._expected_improvement(mu, sigma, y_best)
        
        # Seleccionar mejor
        best_idx = np.argmax(ei)
        best_x = X_candidates[best_idx]
        
        # Decodificar
        config = self.pipeline_space.decode(best_x)
        
        return config.to_dict()
    
    def suggest_initial(self, n: int = 5) -> List[Dict[str, Any]]:
        """Sugiere configuraciones iniciales (warm start)."""
        n_pool = max(n * 20, 100)
        X_pool = self.pipeline_space.sample_random(n_pool)
        X_pool_tensor = torch.tensor(X_pool, dtype=torch.float32).to(self.device)
        
        self.model.eval()
        self.likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.model(X_pool_tensor))
            mu = pred.mean.cpu().numpy()
        
        # Top candidatos
        top_k = min(n * 3, n_pool)
        top_indices = np.argsort(mu)[-top_k:]
        
        # Selección diversa
        selected = [top_indices[-1]]
        for _ in range(n - 1):
            max_min_dist = -1
            best_idx = None
            
            for idx in top_indices:
                if idx in selected:
                    continue
                min_dist = min(np.linalg.norm(X_pool[idx] - X_pool[s]) for s in selected)
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = idx
            
            if best_idx is not None:
                selected.append(best_idx)
        
        configs = [self.pipeline_space.decode(X_pool[i]).to_dict() for i in selected]
        return configs
    
    def observe(self, config: Dict[str, Any], score: float) -> None:
        """Registra una observación."""
        pipeline_config = PipelineConfig.from_dict(config)
        x = self.pipeline_space.encode(pipeline_config)
        
        self.X_observed.append(x)
        self.y_observed.append(score)
        self.configs_observed.append(pipeline_config)
        
        best_so_far = max(self.y_observed)
        self.best_y_history.append(best_so_far)
        
        self._update_gp()
        
        if len(self.y_observed) % self.finetune_frequency == 0:
            self._finetune()
    
    def get_best(self) -> Tuple[Dict[str, Any], float]:
        """Retorna la mejor configuración encontrada."""
        if not self.y_observed:
            raise ValueError("No observations yet")
        
        best_idx = np.argmax(self.y_observed)
        return self.configs_observed[best_idx].to_dict(), self.y_observed[best_idx]
    
    def get_result(self) -> PipelineOptimizationResult:
        """Retorna resultado completo."""
        best_config, best_score = self.get_best()
        
        return PipelineOptimizationResult(
            algorithm=self.algorithm,
            best_config=PipelineConfig.from_dict(best_config),
            best_score=best_score,
            n_evaluations=len(self.y_observed),
            history=self.best_y_history.copy(),
            all_configs=self.configs_observed.copy(),
            all_scores=self.y_observed.copy()
        )
    
    def reset(self) -> None:
        """Reinicia el optimizador."""
        self.X_observed = []
        self.y_observed = []
        self.configs_observed = []
        self.best_y_history = []
        self._n_suggests = 0
    
    def _update_gp(self) -> None:
        """Actualiza datos del GP."""
        if not self.X_observed:
            return
        
        X_tensor = torch.tensor(np.array(self.X_observed), dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(np.array(self.y_observed), dtype=torch.float32).to(self.device)
        
        self.model.set_train_data(X_tensor, y_tensor, strict=False)
    
    def _finetune(self, n_epochs: int = None, lr: float = 1e-4) -> None:
        """Fine-tuning del modelo."""
        if len(self.X_observed) < 2:
            return
        
        if n_epochs is None:
            n_epochs = self.finetune_epochs
        
        self.model.train()
        self.likelihood.train()
        
        X_tensor = torch.tensor(np.array(self.X_observed), dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(np.array(self.y_observed), dtype=torch.float32).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        for _ in range(n_epochs):
            optimizer.zero_grad()
            output = self.model(X_tensor)
            loss = -mll(output, y_tensor)
            loss.backward()
            optimizer.step()
    
    def _expected_improvement(self, mu: np.ndarray, sigma: np.ndarray, y_best: float) -> np.ndarray:
        """Calcula Expected Improvement."""
        sigma = np.maximum(sigma, 1e-8)
        improvement = mu - y_best - self.xi
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma < 1e-8] = 0.0
        return ei


# =============================================================================
# Función de alto nivel
# =============================================================================

def optimize_pipeline(
    algorithm: str,
    evaluation_fn,  # Callable[[Dict], float]
    budget: int = 30,
    n_init: int = 5,
    checkpoint_dir: Optional[str] = None,
    verbose: bool = True
) -> PipelineOptimizationResult:
    """
    Optimiza pipeline completo usando FSBO.
    
    Args:
        algorithm: Nombre del algoritmo
        evaluation_fn: Función que recibe config y retorna score
        budget: Número total de evaluaciones
        n_init: Configuraciones iniciales
        checkpoint_dir: Directorio de checkpoints
        verbose: Mostrar progreso
        
    Returns:
        PipelineOptimizationResult
        
    Example:
        >>> def evaluate(config):
        ...     # Construir pipeline
        ...     pipeline = build_pipeline(
        ...         imputer=config['pipeline']['imputer_strategy'],
        ...         scaler=config['pipeline']['scaler'],
        ...         feature_selection=config['pipeline']['feature_selection'],
        ...         classifier=RandomForestClassifier(**config['classifier'])
        ...     )
        ...     pipeline.fit(X_train, y_train)
        ...     return pipeline.score(X_val, y_val)
        >>> 
        >>> result = optimize_pipeline('random_forest', evaluate, budget=30)
    """
    optimizer = FSBOPipelineOptimizer.from_pretrained(algorithm, checkpoint_dir)
    
    if verbose:
        print(f"\n🎯 Optimizando pipeline: {algorithm}")
        print(f"   Budget: {budget} evaluaciones")
    
    # Warm start
    if verbose:
        print(f"   Warm start: {n_init} configuraciones...")
    
    initial_configs = optimizer.suggest_initial(n_init)
    for i, config in enumerate(initial_configs):
        score = evaluation_fn(config)
        optimizer.observe(config, score)
        if verbose:
            print(f"   [{i+1}/{n_init}] Score: {score:.4f}")
    
    # BO loop
    remaining = budget - n_init
    if verbose:
        print(f"   BO loop ({remaining} iteraciones)...")
    
    for i in range(remaining):
        config = optimizer.suggest()
        score = evaluation_fn(config)
        optimizer.observe(config, score)
        
        if verbose and (i + 1) % 5 == 0:
            _, best_score = optimizer.get_best()
            print(f"   [{n_init + i + 1}/{budget}] Best: {best_score:.4f}")
    
    result = optimizer.get_result()
    
    if verbose:
        print(f"\n   ✅ Completado!")
        print(f"   Mejor score: {result.best_score:.4f}")
        print(f"   Pipeline: {result.best_config.to_dict()['pipeline']}")
        print(f"   Classifier: {result.best_config.to_dict()['classifier']}")
    
    return result


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test FSBOPipelineOptimizer')
    parser.add_argument('--algorithm', type=str, default='adaboost')
    parser.add_argument('--budget', type=int, default=20)
    
    args = parser.parse_args()
    
    print("Testing FSBOPipelineOptimizer...")
    
    # Dummy evaluation
    def dummy_evaluate(config):
        score = 0.7 + np.random.normal(0, 0.05)
        return min(max(score, 0.5), 1.0)
    
    try:
        result = optimize_pipeline(
            algorithm=args.algorithm,
            evaluation_fn=dummy_evaluate,
            budget=args.budget,
            verbose=True
        )
        print(f"\n✅ Test completado!")
    except FileNotFoundError as e:
        print(f"\n⚠️ {e}")
        print("   Entrena primero con: python scripts/train_fsbo_pipeline.py")
