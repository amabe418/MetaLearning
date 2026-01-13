"""
FSBOOptimizer: Clase principal para optimización de hiperparámetros con FSBO.

Esta clase proporciona una API limpia (observe/suggest) para integrar
con el módulo de meta-learning.

Flujo de integración:
    1. Meta-Learning sugiere algoritmos para un dataset
    2. FSBOOptimizer optimiza hiperparámetros de cada algoritmo
    3. Se retornan las configuraciones óptimas

Uso básico:
    optimizer = FSBOOptimizer.from_pretrained('adaboost')
    
    # Loop de optimización
    for _ in range(budget):
        x_next = optimizer.suggest()
        y_next = evaluate_model(x_next)  # Entrenar y evaluar
        optimizer.observe(x_next, y_next)
    
    best_config = optimizer.get_best()

Integración con Meta-Learning:
    from fsbo_optimizer import optimize_algorithms
    
    # Meta-learning sugiere algoritmos
    algorithms = meta_learner.suggest(dataset)  # ['random_forest', 'svm']
    
    # FSBO optimiza cada uno
    results = optimize_algorithms(
        algorithms=algorithms,
        evaluation_fn=lambda alg, hp: train_and_evaluate(dataset, alg, hp),
        budget_per_algorithm=30
    )

Autor: Proyecto académico MetaLearning
Basado en: Wistuba & Grabocka (ICLR 2021) - FSBO
"""

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
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
import json
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Modelos
# =============================================================================

class DeepKernelNetwork(nn.Module):
    """Red neuronal φ que proyecta hiperparámetros a espacio latente."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, n_layers: int = 2):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
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
class OptimizationResult:
    """Resultado de una optimización de hiperparámetros."""
    algorithm: str
    best_config: Dict[str, Any]
    best_score: float
    n_evaluations: int
    history: List[float]
    all_configs: List[Dict[str, Any]]
    all_scores: List[float]


@dataclass
class HyperparameterSpace:
    """Define el espacio de búsqueda de un algoritmo."""
    name: str
    parameters: Dict[str, Dict]  # {param_name: {type, range, ...}}
    use_one_hot: bool = True  # Si usar one-hot encoding para categóricos
    
    def _get_encoded_dim(self) -> int:
        """Calcula la dimensión del vector codificado (con one-hot si aplica)."""
        dim = 0
        for spec in self.parameters.values():
            if spec['type'] == 'categorical' and self.use_one_hot:
                dim += len(spec['choices'])
            else:
                dim += 1
        return dim
    
    def sample_random(self, n: int = 1) -> np.ndarray:
        """Genera n configuraciones aleatorias normalizadas [0,1]."""
        dim = self._get_encoded_dim()
        return np.random.rand(n, dim).astype(np.float32)
    
    def decode(self, x_normalized: np.ndarray) -> Dict[str, Any]:
        """Convierte vector normalizado a diccionario de hiperparámetros."""
        config = {}
        idx = 0  # Índice en el vector x_normalized
        
        for name, spec in self.parameters.items():
            if spec['type'] == 'float':
                val = x_normalized[idx]
                lo, hi = spec['range']
                if spec.get('log', False):
                    config[name] = float(np.exp(np.log(lo) + val * (np.log(hi) - np.log(lo))))
                else:
                    config[name] = float(lo + val * (hi - lo))
                idx += 1
                    
            elif spec['type'] == 'int':
                val = x_normalized[idx]
                lo, hi = spec['range']
                config[name] = int(np.round(lo + val * (hi - lo)))
                idx += 1
                
            elif spec['type'] == 'categorical':
                choices = spec['choices']
                
                if self.use_one_hot:
                    # Decodificar one-hot
                    n_choices = len(choices)
                    one_hot_values = x_normalized[idx:idx+n_choices]
                    choice_idx = int(np.argmax(one_hot_values))
                    idx += n_choices
                else:
                    # Decodificado simple
                    val = x_normalized[idx]
                    choice_idx = int(val * (len(choices) - 1)) if len(choices) > 1 else 0
                    choice_idx = max(0, min(choice_idx, len(choices) - 1))
                    idx += 1
                
                choice = choices[choice_idx]
                
                # Convertir strings de booleanos de vuelta a bool si es necesario
                if choice in ['True', 'true']:
                    config[name] = True
                elif choice in ['False', 'false']:
                    config[name] = False
                else:
                    config[name] = choice
                
        return config
    
    def encode(self, config: Dict[str, Any]) -> np.ndarray:
        """Convierte diccionario de hiperparámetros a vector normalizado."""
        x_list = []
        
        for name, spec in self.parameters.items():
            val = config.get(name)
            if val is None:
                # Si falta un valor, usar valores por defecto
                if spec['type'] == 'categorical' and self.use_one_hot:
                    x_list.extend([0.0] * len(spec['choices']))
                else:
                    x_list.append(0.5)
                continue
            
            try:
                if spec['type'] == 'float':
                    lo, hi = spec['range']
                    val = float(val)
                    if spec.get('log', False):
                        encoded = (np.log(val) - np.log(lo)) / (np.log(hi) - np.log(lo))
                    else:
                        encoded = (val - lo) / (hi - lo)
                    x_list.append(encoded)
                        
                elif spec['type'] == 'int':
                    lo, hi = spec['range']
                    val = int(val) if not isinstance(val, (int, float)) else val
                    encoded = (val - lo) / (hi - lo)
                    x_list.append(encoded)
                    
                elif spec['type'] == 'categorical':
                    choices = spec['choices']
                    val_str = str(val)
                    
                    # Encontrar el índice del valor
                    if val in choices:
                        idx = choices.index(val)
                    elif val_str in choices:
                        idx = choices.index(val_str)
                    else:
                        logger.warning(f"Valor '{val}' no encontrado en choices para {name}, usando default")
                        idx = 0
                    
                    if self.use_one_hot:
                        # One-hot encoding
                        one_hot = [0.0] * len(choices)
                        one_hot[idx] = 1.0
                        x_list.extend(one_hot)
                    else:
                        # Encoding simple
                        x_list.append(idx / max(len(choices) - 1, 1))
                        
            except (ValueError, TypeError) as e:
                logger.error(f"Error codificando {name}={val} (tipo={spec['type']}): {e}")
                if spec['type'] == 'categorical' and self.use_one_hot:
                    x_list.extend([0.0] * len(spec['choices']))
                else:
                    x_list.append(0.5)
        
        x = np.array(x_list, dtype=np.float32)
        return np.clip(x, 0, 1)


# =============================================================================
# Espacios de hiperparámetros predefinidos
# =============================================================================

HYPERPARAMETER_SPACES = {
    'adaboost': HyperparameterSpace(
        name='adaboost',
        parameters={
            'n_estimators': {'type': 'int', 'range': [50, 500]},
            'learning_rate': {'type': 'float', 'range': [0.01, 2.0], 'log': True},
            'algorithm': {'type': 'categorical', 'choices': ['SAMME', 'SAMME.R']},
        }
    ),
    'random_forest': HyperparameterSpace(
        name='random_forest',
        parameters={
            'n_estimators': {'type': 'int', 'range': [50, 500]},
            'max_depth': {'type': 'int', 'range': [2, 50]},
            'min_samples_split': {'type': 'int', 'range': [2, 20]},
            'min_samples_leaf': {'type': 'int', 'range': [1, 10]},
        }
    ),
    'svm': HyperparameterSpace(
        name='svm',
        parameters={
            'C': {'type': 'float', 'range': [1e-3, 1e3], 'log': True},
            'gamma': {'type': 'float', 'range': [1e-4, 1e1], 'log': True},
            'kernel': {'type': 'categorical', 'choices': ['rbf', 'poly', 'sigmoid']},
        }
    ),
    'libsvm_svc': HyperparameterSpace(
        name='libsvm_svc',
        parameters={
            'C': {'type': 'float', 'range': [1e-3, 1e3], 'log': True},
            'gamma': {'type': 'float', 'range': [1e-4, 1e1], 'log': True},
            'kernel': {'type': 'categorical', 'choices': ['rbf', 'poly', 'sigmoid']},
        }
    ),
    'gradient_boosting': HyperparameterSpace(
        name='gradient_boosting',
        parameters={
            'n_estimators': {'type': 'int', 'range': [50, 500]},
            'learning_rate': {'type': 'float', 'range': [0.01, 0.5], 'log': True},
            'max_depth': {'type': 'int', 'range': [2, 10]},
            'subsample': {'type': 'float', 'range': [0.5, 1.0]},
        }
    ),
}


# =============================================================================
# FSBOOptimizer - Clase Principal
# =============================================================================

class FSBOOptimizer:
    """
    Optimizador de hiperparámetros basado en FSBO.
    
    Proporciona una API observe/suggest para integración con meta-learning.
    
    Attributes:
        algorithm: Nombre del algoritmo a optimizar
        hp_space: Espacio de hiperparámetros
        model: Deep Kernel GP pre-entrenado
        likelihood: Likelihood del GP
        X_observed: Configuraciones observadas (normalizadas)
        y_observed: Scores observados
        
    Example:
        >>> optimizer = FSBOOptimizer.from_pretrained('random_forest')
        >>> 
        >>> # Warm start
        >>> initial_configs = optimizer.suggest_initial(n=5)
        >>> for config in initial_configs:
        ...     score = train_and_evaluate(dataset, 'random_forest', config)
        ...     optimizer.observe(config, score)
        >>> 
        >>> # BO loop
        >>> for _ in range(25):
        ...     config = optimizer.suggest()
        ...     score = train_and_evaluate(dataset, 'random_forest', config)
        ...     optimizer.observe(config, score)
        >>> 
        >>> best = optimizer.get_best()
    """
    
    def __init__(
        self,
        algorithm: str,
        model: DeepKernelGP,
        likelihood: GaussianLikelihood,
        hp_space: HyperparameterSpace,
        input_dim: int,
        device: str = 'cpu'
    ):
        self.algorithm = algorithm
        self.model = model
        self.likelihood = likelihood
        self.hp_space = hp_space
        self.input_dim = input_dim
        self.device = device
        
        # Observaciones
        self.X_observed: List[np.ndarray] = []
        self.y_observed: List[float] = []
        self.configs_observed: List[Dict] = []
        
        # Historial
        self.best_y_history: List[float] = []
        
        # Configuración
        self.xi = 0.01  # Factor de exploración para EI
        self.finetune_frequency = 5
        self.finetune_epochs = 20
        self._n_suggests = 0
        
    @staticmethod
    def _load_configspace_from_json(
        algorithm: str,
        configspace_dir: Path
    ) -> HyperparameterSpace:
        """
        Carga el espacio de hiperparámetros desde un archivo JSON.
        
        Args:
            algorithm: Nombre del algoritmo
            configspace_dir: Directorio con los archivos JSON
            
        Returns:
            HyperparameterSpace configurado
        """
        # Asegurar que configspace_dir es Path
        if not isinstance(configspace_dir, Path):
            configspace_dir = Path(configspace_dir)
        
        # Buscar el archivo JSON correspondiente
        json_path = configspace_dir / f"{algorithm}_configspace.json"
        
        logger.info(f"Buscando configspace en: {json_path}")
        
        if not json_path.exists():
            logger.warning(f"No se encontró configspace JSON para {algorithm} en {json_path}, usando espacio predefinido")
            # Fallback al espacio predefinido si existe
            if algorithm in HYPERPARAMETER_SPACES:
                return HYPERPARAMETER_SPACES[algorithm]
            else:
                raise FileNotFoundError(f"No configspace disponible para {algorithm}")
        
        # Leer el JSON
        with open(json_path, 'r') as f:
            config_data = json.load(f)
        
        # Convertir hiperparámetros al formato HyperparameterSpace
        parameters = {}
        
        for hp in config_data['hyperparameters']:
            name = hp['name']
            hp_type = hp['type']
            
            # Ignorar hiperparámetros constantes (no se optimizan)
            if hp_type == 'constant':
                continue
            
            # Convertir según el tipo
            if hp_type == 'uniform_float':
                parameters[name] = {
                    'type': 'float',
                    'range': [hp['lower'], hp['upper']],
                    'log': hp.get('log', False)
                }
            
            elif hp_type == 'uniform_int':
                parameters[name] = {
                    'type': 'int',
                    'range': [hp['lower'], hp['upper']],
                    'log': hp.get('log', False)
                }
            
            elif hp_type == 'categorical':
                # Convertir booleanos a strings si es necesario
                choices = hp['choices']
                choices = [str(c) if isinstance(c, bool) else c for c in choices]
                parameters[name] = {
                    'type': 'categorical',
                    'choices': choices
                }
            
            else:
                logger.warning(f"Tipo de hiperparámetro desconocido: {hp_type} para {name}")
        
        logger.info(f"Cargado configspace para {algorithm}: {len(parameters)} hiperparámetros")
        
        return HyperparameterSpace(
            name=algorithm,
            parameters=parameters
        )
    
    @classmethod
    def from_pretrained(
        cls,
        algorithm: str,
        checkpoint_dir: Optional[str] = None,
        configspace_dir: Optional[str] = None,
        device: str = 'cpu'
    ) -> 'FSBOOptimizer':
        """
        Carga un optimizador desde un checkpoint pre-entrenado.
        
        Args:
            algorithm: Nombre del algoritmo ('adaboost', 'random_forest', etc.)
            checkpoint_dir: Directorio de checkpoints (opcional)
            configspace_dir: Directorio de configspaces JSON (opcional)
            device: Dispositivo ('cpu' o 'cuda')
            
        Returns:
            FSBOOptimizer configurado y listo para usar
        """
        if checkpoint_dir is None:
            checkpoint_dir = Path(__file__).parent.parent / 'experiments' / 'checkpoints'
        else:
            checkpoint_dir = Path(checkpoint_dir)
        
        if configspace_dir is None:
            configspace_dir = Path(__file__).parent.parent / 'data' / 'configspace'
        else:
            configspace_dir = Path(configspace_dir)
        
        # Buscar checkpoint
        checkpoints = list(checkpoint_dir.glob(f'fsbo_{algorithm}_*.pt'))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint found for {algorithm}")
        
        checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
        
        # Cargar checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint.get('config', {})
        hidden_dim = config.get('hidden_dim', 128)
        
        # Determinar input_dim del checkpoint
        # Intentar inferir del state_dict
        state_dict = checkpoint['model_state']
        first_layer_key = 'feature_extractor.network.0.weight'
        if first_layer_key in state_dict:
            input_dim = state_dict[first_layer_key].shape[1]
        else:
            # Fallback
            input_dim = 8
        
        # Crear modelo
        train_x = torch.zeros(1, input_dim).to(device)
        train_y = torch.zeros(1).to(device)
        
        feature_extractor = DeepKernelNetwork(input_dim, hidden_dim).to(device)
        likelihood = GaussianLikelihood().to(device)
        model = DeepKernelGP(train_x, train_y, likelihood, feature_extractor).to(device)
        
        # Cargar pesos
        model.load_state_dict(checkpoint['model_state'])
        likelihood.load_state_dict(checkpoint['likelihood_state'])
        
        # Cargar espacio de hiperparámetros real desde JSON
        hp_space = cls._load_configspace_from_json(algorithm, configspace_dir)
        
        # VALIDACIÓN: verificar compatibilidad de dimensiones
        n_params = len(hp_space.parameters)
        encoded_dim = hp_space._get_encoded_dim()
        
        if encoded_dim != input_dim:
            logger.warning(
                f"⚠️ Mismatch de dimensiones para {algorithm}:\n"
                f"   - Modelo entrenado: {input_dim} dimensiones\n"
                f"   - ConfigSpace: {n_params} hiperparámetros → {encoded_dim} dimensiones (con one-hot)\n"
                f"   El modelo puede no funcionar correctamente."
            )
            # Intentar ajustar el one_hot encoding
            if input_dim < encoded_dim:
                logger.info("   Deshabilitando one-hot encoding...")
                hp_space.use_one_hot = False
                encoded_dim = hp_space._get_encoded_dim()
                if encoded_dim != input_dim:
                    logger.error(f"   Aún hay mismatch: {encoded_dim} != {input_dim}")
        
        logger.info(f"Loaded FSBO optimizer for {algorithm} from {checkpoint_path.name}")
        logger.info(f"ConfigSpace: {n_params} hiperparámetros → {encoded_dim} dimensiones")
        
        return cls(
            algorithm=algorithm,
            model=model,
            likelihood=likelihood,
            hp_space=hp_space,
            input_dim=input_dim,
            device=device
        )
    
    def observe(self, config: Dict[str, Any], score: float) -> None:
        """
        Registra una nueva observación (configuración evaluada).
        
        Args:
            config: Diccionario de hiperparámetros
            score: Score obtenido (accuracy, AUC, etc.)
        """
        # Convertir config a vector normalizado
        x = self.hp_space.encode(config)
        
        self.X_observed.append(x)
        self.y_observed.append(score)
        self.configs_observed.append(config.copy())
        
        # Actualizar historial de mejor
        best_so_far = max(self.y_observed)
        self.best_y_history.append(best_so_far)
        
        # Actualizar GP
        self._update_gp()
        
        # Fine-tuning periódico
        if len(self.y_observed) % self.finetune_frequency == 0:
            self._finetune()
    
    def observe_batch(self, configs: List[Dict], scores: List[float]) -> None:
        """Registra múltiples observaciones."""
        for config, score in zip(configs, scores):
            x = self.hp_space.encode(config)
            self.X_observed.append(x)
            self.y_observed.append(score)
            self.configs_observed.append(config.copy())
        
        # Actualizar historial
        for i in range(len(configs)):
            best_so_far = max(self.y_observed[:len(self.y_observed) - len(configs) + i + 1])
            self.best_y_history.append(best_so_far)
        
        self._update_gp()
        self._finetune()
    
    def suggest(self, n_candidates: int = 1000) -> Dict[str, Any]:
        """
        Sugiere la siguiente configuración a evaluar.
        
        Usa Expected Improvement como función de adquisición.
        
        Args:
            n_candidates: Número de candidatos a evaluar
            
        Returns:
            Diccionario con la configuración sugerida
        """
        self._n_suggests += 1
        
        # Generar candidatos aleatorios
        X_candidates = self.hp_space.sample_random(n_candidates)
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
        
        # Seleccionar mejor candidato
        best_idx = np.argmax(ei)
        best_x = X_candidates[best_idx]
        
        # Decodificar a configuración
        config = self.hp_space.decode(best_x)
        
        return config
    
    def suggest_initial(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Sugiere configuraciones iniciales (warm start).
        
        Usa el modelo pre-entrenado para seleccionar configuraciones
        prometedoras y diversas.
        
        Args:
            n: Número de configuraciones iniciales
            
        Returns:
            Lista de configuraciones
        """
        # Generar pool de candidatos
        n_pool = max(n * 20, 100)
        X_pool = self.hp_space.sample_random(n_pool)
        X_pool_tensor = torch.tensor(X_pool, dtype=torch.float32).to(self.device)
        
        # Predecir con modelo pre-entrenado
        self.model.eval()
        self.likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.model(X_pool_tensor))
            mu = pred.mean.cpu().numpy()
        
        # Seleccionar top candidatos
        top_k = min(n * 3, n_pool)
        top_indices = np.argsort(mu)[-top_k:]
        
        # De los top, seleccionar diversos (maximizar distancia mínima)
        selected = [top_indices[-1]]  # Empezar con el mejor
        
        for _ in range(n - 1):
            max_min_dist = -1
            best_idx = None
            
            for idx in top_indices:
                if idx in selected:
                    continue
                
                # Distancia mínima a los ya seleccionados
                min_dist = min(
                    np.linalg.norm(X_pool[idx] - X_pool[s])
                    for s in selected
                )
                
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = idx
            
            if best_idx is not None:
                selected.append(best_idx)
        
        # Decodificar configuraciones
        configs = [self.hp_space.decode(X_pool[i]) for i in selected]
        
        return configs
    
    def get_best(self) -> Tuple[Dict[str, Any], float]:
        """
        Retorna la mejor configuración encontrada.
        
        Returns:
            (config, score) - Mejor configuración y su score
        """
        if not self.y_observed:
            raise ValueError("No observations yet")
        
        best_idx = np.argmax(self.y_observed)
        return self.configs_observed[best_idx], self.y_observed[best_idx]
    
    def get_result(self) -> OptimizationResult:
        """Retorna resultado completo de la optimización."""
        best_config, best_score = self.get_best()
        
        return OptimizationResult(
            algorithm=self.algorithm,
            best_config=best_config,
            best_score=best_score,
            n_evaluations=len(self.y_observed),
            history=self.best_y_history.copy(),
            all_configs=self.configs_observed.copy(),
            all_scores=self.y_observed.copy()
        )
    
    def reset(self) -> None:
        """Reinicia el optimizador para una nueva tarea."""
        self.X_observed = []
        self.y_observed = []
        self.configs_observed = []
        self.best_y_history = []
        self._n_suggests = 0
    
    # =========================================================================
    # Métodos privados
    # =========================================================================
    
    def _update_gp(self) -> None:
        """Actualiza los datos del GP."""
        if not self.X_observed:
            return
        
        X_tensor = torch.tensor(
            np.array(self.X_observed), 
            dtype=torch.float32
        ).to(self.device)
        
        y_tensor = torch.tensor(
            np.array(self.y_observed), 
            dtype=torch.float32
        ).to(self.device)
        
        self.model.set_train_data(X_tensor, y_tensor, strict=False)
    
    def _finetune(self, n_epochs: int = None, lr: float = 1e-4) -> None:
        """Fine-tuning del modelo en las observaciones actuales."""
        if len(self.X_observed) < 2:
            return
        
        if n_epochs is None:
            n_epochs = self.finetune_epochs
        
        self.model.train()
        self.likelihood.train()
        
        X_tensor = torch.tensor(
            np.array(self.X_observed), 
            dtype=torch.float32
        ).to(self.device)
        
        y_tensor = torch.tensor(
            np.array(self.y_observed), 
            dtype=torch.float32
        ).to(self.device)
        
        optimizer = torch.optim.Adam([
            {'params': self.model.feature_extractor.parameters(), 'lr': lr},
            {'params': self.model.covar_module.parameters(), 'lr': lr},
            {'params': self.model.mean_module.parameters(), 'lr': lr},
            {'params': self.likelihood.parameters(), 'lr': lr},
        ])
        
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        for _ in range(n_epochs):
            optimizer.zero_grad()
            output = self.model(X_tensor)
            loss = -mll(output, y_tensor)
            loss.backward()
            optimizer.step()
    
    def _expected_improvement(
        self, 
        mu: np.ndarray, 
        sigma: np.ndarray, 
        y_best: float
    ) -> np.ndarray:
        """Calcula Expected Improvement."""
        sigma = np.maximum(sigma, 1e-8)
        improvement = mu - y_best - self.xi
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma < 1e-8] = 0.0
        return ei


# =============================================================================
# Función de alto nivel para integración con Meta-Learning
# =============================================================================

def optimize_algorithm(
    algorithm: str,
    evaluation_fn: Callable[[Dict[str, Any]], float],
    budget: int = 30,
    n_init: int = 5,
    checkpoint_dir: Optional[str] = None,
    verbose: bool = True
) -> OptimizationResult:
    """
    Optimiza hiperparámetros de un algoritmo usando FSBO.
    
    Esta función es el punto de entrada principal para integración
    con el módulo de meta-learning.
    
    Args:
        algorithm: Nombre del algoritmo ('adaboost', 'random_forest', etc.)
        evaluation_fn: Función que recibe config y retorna score
                      def evaluation_fn(config: Dict) -> float
        budget: Número total de evaluaciones
        n_init: Configuraciones iniciales (warm start)
        checkpoint_dir: Directorio de checkpoints FSBO
        verbose: Mostrar progreso
        
    Returns:
        OptimizationResult con la mejor configuración encontrada
        
    Example:
        >>> def evaluate(config):
        ...     model = RandomForestClassifier(**config)
        ...     model.fit(X_train, y_train)
        ...     return model.score(X_val, y_val)
        >>> 
        >>> result = optimize_algorithm(
        ...     algorithm='random_forest',
        ...     evaluation_fn=evaluate,
        ...     budget=30
        ... )
        >>> print(f"Best config: {result.best_config}")
        >>> print(f"Best score: {result.best_score}")
    """
    # Cargar optimizador
    try:
        optimizer = FSBOOptimizer.from_pretrained(algorithm, checkpoint_dir)
    except FileNotFoundError:
        logger.warning(f"No pretrained model for {algorithm}, using random search")
        # Fallback a búsqueda aleatoria si no hay checkpoint
        return _random_search_fallback(algorithm, evaluation_fn, budget)
    
    if verbose:
        print(f"\n🎯 Optimizando: {algorithm}")
        print(f"   Budget: {budget} evaluaciones")
    
    # Warm start
    if verbose:
        print(f"   Warm start: {n_init} configuraciones iniciales...")
    
    initial_configs = optimizer.suggest_initial(n_init)
    
    for i, config in enumerate(initial_configs):
        score = evaluation_fn(config)
        optimizer.observe(config, score)
        
        if verbose:
            print(f"   [{i+1}/{n_init}] Score: {score:.4f}")
    
    # BO loop
    remaining = budget - n_init
    
    if verbose:
        print(f"   Ejecutando BO loop ({remaining} iteraciones)...")
    
    for i in range(remaining):
        config = optimizer.suggest()
        score = evaluation_fn(config)
        optimizer.observe(config, score)
        
        if verbose and (i + 1) % 5 == 0:
            best_config, best_score = optimizer.get_best()
            print(f"   [{n_init + i + 1}/{budget}] Best: {best_score:.4f}")
    
    result = optimizer.get_result()
    
    if verbose:
        print(f"\n   ✅ Completado!")
        print(f"   Mejor score: {result.best_score:.4f}")
        print(f"   Mejor config: {result.best_config}")
    
    return result


def optimize_algorithms(
    algorithms: List[str],
    evaluation_fn: Callable[[str, Dict[str, Any]], float],
    budget_per_algorithm: int = 30,
    n_init: int = 5,
    checkpoint_dir: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, OptimizationResult]:
    """
    Optimiza hiperparámetros de múltiples algoritmos.
    
    Función principal para integración con meta-learning cuando
    se sugieren múltiples algoritmos para un dataset.
    
    Args:
        algorithms: Lista de algoritmos a optimizar
        evaluation_fn: Función que recibe (algorithm, config) y retorna score
                      def evaluation_fn(alg: str, config: Dict) -> float
        budget_per_algorithm: Evaluaciones por algoritmo
        n_init: Configuraciones iniciales por algoritmo
        checkpoint_dir: Directorio de checkpoints FSBO
        verbose: Mostrar progreso
        
    Returns:
        Dict[algorithm_name, OptimizationResult]
        
    Example:
        >>> # Meta-learning sugiere algoritmos
        >>> suggested_algorithms = ['random_forest', 'adaboost', 'svm']
        >>> 
        >>> def evaluate(algorithm, config):
        ...     if algorithm == 'random_forest':
        ...         model = RandomForestClassifier(**config)
        ...     elif algorithm == 'adaboost':
        ...         model = AdaBoostClassifier(**config)
        ...     # ... etc
        ...     model.fit(X_train, y_train)
        ...     return model.score(X_val, y_val)
        >>> 
        >>> results = optimize_algorithms(
        ...     algorithms=suggested_algorithms,
        ...     evaluation_fn=evaluate,
        ...     budget_per_algorithm=30
        ... )
        >>> 
        >>> # Encontrar el mejor algoritmo
        >>> best_alg = max(results, key=lambda a: results[a].best_score)
        >>> print(f"Best algorithm: {best_alg}")
        >>> print(f"Best config: {results[best_alg].best_config}")
    """
    results = {}
    
    if verbose:
        print("=" * 60)
        print("🚀 FSBO Multi-Algorithm Optimization")
        print("=" * 60)
        print(f"\nAlgoritmos: {algorithms}")
        print(f"Budget por algoritmo: {budget_per_algorithm}")
    
    for algorithm in algorithms:
        # Crear función de evaluación específica para este algoritmo
        alg_eval_fn = lambda config, alg=algorithm: evaluation_fn(alg, config)
        
        result = optimize_algorithm(
            algorithm=algorithm,
            evaluation_fn=alg_eval_fn,
            budget=budget_per_algorithm,
            n_init=n_init,
            checkpoint_dir=checkpoint_dir,
            verbose=verbose
        )
        
        results[algorithm] = result
    
    if verbose:
        print("\n" + "=" * 60)
        print("📋 RESUMEN")
        print("=" * 60)
        
        for alg, res in sorted(results.items(), key=lambda x: -x[1].best_score):
            print(f"\n{alg}:")
            print(f"  Score: {res.best_score:.4f}")
            print(f"  Config: {res.best_config}")
        
        best_alg = max(results, key=lambda a: results[a].best_score)
        print(f"\n🏆 Mejor algoritmo: {best_alg} (score={results[best_alg].best_score:.4f})")
    
    return results


def _random_search_fallback(
    algorithm: str,
    evaluation_fn: Callable[[Dict], float],
    budget: int
) -> OptimizationResult:
    """Fallback a búsqueda aleatoria si no hay checkpoint."""
    hp_space = HYPERPARAMETER_SPACES.get(algorithm)
    
    if hp_space is None:
        hp_space = HyperparameterSpace(
            name=algorithm,
            parameters={f'hp_{i}': {'type': 'float', 'range': [0, 1]} for i in range(5)}
        )
    
    configs = []
    scores = []
    
    for _ in range(budget):
        x = hp_space.sample_random(1)[0]
        config = hp_space.decode(x)
        score = evaluation_fn(config)
        configs.append(config)
        scores.append(score)
    
    best_idx = np.argmax(scores)
    
    return OptimizationResult(
        algorithm=algorithm,
        best_config=configs[best_idx],
        best_score=scores[best_idx],
        n_evaluations=budget,
        history=[max(scores[:i+1]) for i in range(len(scores))],
        all_configs=configs,
        all_scores=scores
    )


# =============================================================================
# CLI para testing
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test FSBOOptimizer')
    parser.add_argument('--algorithm', type=str, default='adaboost')
    parser.add_argument('--budget', type=int, default=20)
    
    args = parser.parse_args()
    
    print("Testing FSBOOptimizer...")
    
    # Función de evaluación dummy (simula entrenamiento)
    def dummy_evaluate(config):
        # Simular que algunas configuraciones son mejores
        # Config tiene formato {'hp_0': val, 'hp_1': val, ...}
        score = 0.7
        
        # Simular que valores intermedios son mejores
        for key, val in config.items():
            if 0.3 < val < 0.7:
                score += 0.03
        
        score += np.random.normal(0, 0.02)  # Ruido
        return min(max(score, 0.5), 1.0)
    
    result = optimize_algorithm(
        algorithm=args.algorithm,
        evaluation_fn=dummy_evaluate,
        budget=args.budget,
        verbose=True
    )
    
    print(f"\n✅ Test completado!")
    print(f"   Mejor score: {result.best_score:.4f}")

