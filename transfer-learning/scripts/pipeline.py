"""
Pipeline de Integración: Meta-Learning + FSBO Transfer Learning

Este módulo implementa el pipeline completo que:
1. Recibe un dataset y algoritmos sugeridos por meta-learning
2. Optimiza hiperparámetros de cada algoritmo con FSBO
3. Retorna la mejor configuración para el dataset

Mejoras implementadas:
- Warm start inteligente con meta-features del dataset
- Ajuste dinámico de presupuesto por algoritmo
- Transfer de hiperparámetros desde tareas similares

Flujo:
    Dataset → [Meta-Learning] → Algoritmos sugeridos
                                        ↓
    Dataset + Algoritmos → [FSBO Pipeline] → Configuraciones óptimas

Autor: Proyecto académico MetaLearning
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

# Importar FSBOOptimizer
from fsbo_optimizer import (
    FSBOOptimizer, 
    OptimizationResult,
    HyperparameterSpace,
    HYPERPARAMETER_SPACES
)

logger = logging.getLogger(__name__)


# =============================================================================
# Estructuras de datos
# =============================================================================

@dataclass
class DatasetMetaFeatures:
    """
    Meta-features de un dataset para warm start inteligente.
    
    Basado en características comunes en meta-learning:
    - Características simples: n_samples, n_features, n_classes
    - Características estadísticas: mean, std, skewness
    - Características de información: entropy, mutual_info
    """
    dataset_id: str
    n_samples: int
    n_features: int
    n_classes: int
    
    # Estadísticas de features
    feature_means: Optional[np.ndarray] = None
    feature_stds: Optional[np.ndarray] = None
    
    # Ratios útiles
    samples_per_feature: float = 0.0
    samples_per_class: float = 0.0
    class_imbalance: float = 0.0  # 0 = balanceado, 1 = muy desbalanceado
    
    # Vector de meta-features para comparación
    meta_vector: np.ndarray = field(default_factory=lambda: np.array([]))
    
    @classmethod
    def from_data(cls, X: np.ndarray, y: np.ndarray, dataset_id: str = "unknown") -> 'DatasetMetaFeatures':
        """Extrae meta-features de datos."""
        n_samples, n_features = X.shape
        classes, counts = np.unique(y, return_counts=True)
        n_classes = len(classes)
        
        # Calcular estadísticas
        feature_means = np.mean(X, axis=0)
        feature_stds = np.std(X, axis=0)
        
        # Ratios
        samples_per_feature = n_samples / max(n_features, 1)
        samples_per_class = n_samples / max(n_classes, 1)
        
        # Class imbalance (Gini)
        proportions = counts / counts.sum()
        class_imbalance = 1 - np.sum(proportions ** 2)
        
        # Vector de meta-features (normalizado para comparación)
        meta_vector = np.array([
            np.log(n_samples + 1),
            np.log(n_features + 1),
            np.log(n_classes + 1),
            np.log(samples_per_feature + 1),
            np.log(samples_per_class + 1),
            class_imbalance,
            np.mean(feature_means),
            np.mean(feature_stds),
        ])
        
        return cls(
            dataset_id=dataset_id,
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            feature_means=feature_means,
            feature_stds=feature_stds,
            samples_per_feature=samples_per_feature,
            samples_per_class=samples_per_class,
            class_imbalance=class_imbalance,
            meta_vector=meta_vector
        )
    
    def similarity_to(self, other: 'DatasetMetaFeatures') -> float:
        """Calcula similitud con otro dataset (0-1, mayor = más similar)."""
        if len(self.meta_vector) == 0 or len(other.meta_vector) == 0:
            return 0.0
        
        # Distancia euclidiana normalizada
        dist = np.linalg.norm(self.meta_vector - other.meta_vector)
        similarity = 1 / (1 + dist)
        return similarity


@dataclass
class AlgorithmSuggestion:
    """Sugerencia de algoritmo desde meta-learning."""
    name: str
    confidence: float  # 0-1, qué tan seguro está el meta-learner
    expected_performance: float  # Performance estimada
    priority: int = 0  # Para ordenar
    
    
@dataclass
class PipelineResult:
    """Resultado completo del pipeline."""
    dataset_id: str
    meta_features: DatasetMetaFeatures
    algorithm_results: Dict[str, OptimizationResult]
    best_algorithm: str
    best_config: Dict[str, Any]
    best_score: float
    total_evaluations: int
    total_time_seconds: float
    budget_allocation: Dict[str, int]


# =============================================================================
# Base de conocimiento para transfer
# =============================================================================

class KnowledgeBase:
    """
    Base de conocimiento con historial de optimizaciones.
    
    Permite:
    - Almacenar resultados de optimizaciones anteriores
    - Buscar tareas similares por meta-features
    - Transferir hiperparámetros de tareas similares
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else None
        self.entries: List[Dict] = []
        self._load()
    
    def _load(self):
        """Carga base de conocimiento desde disco."""
        if self.storage_path and self.storage_path.exists():
            with open(self.storage_path, 'r') as f:
                self.entries = json.load(f)
            logger.info(f"Loaded {len(self.entries)} entries from knowledge base")
    
    def _save(self):
        """Guarda base de conocimiento a disco."""
        if self.storage_path:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump(self.entries, f, indent=2, default=str)
    
    def add_entry(
        self,
        dataset_meta: DatasetMetaFeatures,
        algorithm: str,
        best_config: Dict[str, Any],
        best_score: float
    ):
        """Añade una entrada a la base de conocimiento."""
        entry = {
            'dataset_id': dataset_meta.dataset_id,
            'meta_vector': dataset_meta.meta_vector.tolist(),
            'algorithm': algorithm,
            'best_config': best_config,
            'best_score': best_score,
            'timestamp': datetime.now().isoformat()
        }
        self.entries.append(entry)
        self._save()
    
    def find_similar_configs(
        self,
        dataset_meta: DatasetMetaFeatures,
        algorithm: str,
        top_k: int = 5
    ) -> List[Tuple[Dict[str, Any], float, float]]:
        """
        Encuentra configuraciones de tareas similares.
        
        Returns:
            Lista de (config, score, similarity)
        """
        results = []
        
        for entry in self.entries:
            if entry['algorithm'] != algorithm:
                continue
            
            # Calcular similitud
            entry_vector = np.array(entry['meta_vector'])
            dist = np.linalg.norm(dataset_meta.meta_vector - entry_vector)
            similarity = 1 / (1 + dist)
            
            results.append((
                entry['best_config'],
                entry['best_score'],
                similarity
            ))
        
        # Ordenar por similitud
        results.sort(key=lambda x: -x[2])
        
        return results[:top_k]


# =============================================================================
# MEJORA 1: Warm Start Inteligente con Meta-Features
# =============================================================================

class IntelligentWarmStart:
    """
    Warm start inteligente que usa meta-features del dataset.
    
    Estrategia:
    1. Buscar tareas similares en la base de conocimiento
    2. Usar sus mejores configuraciones como punto de partida
    3. Añadir exploración diversa alrededor de esas configuraciones
    """
    
    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base
    
    def get_initial_configs(
        self,
        dataset_meta: DatasetMetaFeatures,
        algorithm: str,
        optimizer: FSBOOptimizer,
        n_init: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Genera configuraciones iniciales inteligentes.
        
        Args:
            dataset_meta: Meta-features del dataset objetivo
            algorithm: Algoritmo a optimizar
            optimizer: FSBOOptimizer para generar configs
            n_init: Número de configuraciones
            
        Returns:
            Lista de configuraciones iniciales
        """
        configs = []
        
        # 1. Buscar en base de conocimiento
        similar_configs = self.kb.find_similar_configs(
            dataset_meta, algorithm, top_k=n_init // 2
        )
        
        if similar_configs:
            logger.info(f"Found {len(similar_configs)} similar configurations")
            
            for config, score, similarity in similar_configs:
                if len(configs) >= n_init // 2:
                    break
                    
                # Añadir config transferida (con pequeña perturbación)
                perturbed = self._perturb_config(config, optimizer.hp_space)
                configs.append(perturbed)
                
                logger.debug(f"Transferred config (sim={similarity:.3f}, score={score:.4f})")
        
        # 2. Completar con sugerencias del modelo pre-entrenado
        remaining = n_init - len(configs)
        if remaining > 0:
            model_suggestions = optimizer.suggest_initial(remaining)
            configs.extend(model_suggestions)
        
        return configs[:n_init]
    
    def _perturb_config(
        self, 
        config: Dict[str, Any], 
        hp_space: HyperparameterSpace,
        noise_scale: float = 0.1
    ) -> Dict[str, Any]:
        """Añade pequeña perturbación a una configuración."""
        perturbed = {}
        
        for key, value in config.items():
            if isinstance(value, (int, float)):
                # Añadir ruido gaussiano
                noise = np.random.normal(0, noise_scale * abs(value + 0.1))
                perturbed[key] = value + noise
            else:
                perturbed[key] = value
        
        return perturbed


# =============================================================================
# MEJORA 2: Ajuste Dinámico de Presupuesto
# =============================================================================

class DynamicBudgetAllocator:
    """
    Asigna presupuesto dinámicamente basado en:
    - Confianza del meta-learner en cada algoritmo
    - Progreso de la optimización (early stopping si converge)
    - Dificultad estimada del espacio de hiperparámetros
    """
    
    def __init__(
        self,
        total_budget: int,
        min_budget_per_algorithm: int = 10,
        confidence_weight: float = 0.6,
        complexity_weight: float = 0.4
    ):
        self.total_budget = total_budget
        self.min_budget = min_budget_per_algorithm
        self.confidence_weight = confidence_weight
        self.complexity_weight = complexity_weight
    
    def allocate(
        self,
        suggestions: List[AlgorithmSuggestion]
    ) -> Dict[str, int]:
        """
        Asigna presupuesto a cada algoritmo.
        
        Estrategia:
        - Más presupuesto a algoritmos con mayor confianza
        - Más presupuesto a algoritmos con espacios más complejos
        - Garantizar mínimo para todos
        
        Returns:
            Dict[algorithm_name, budget]
        """
        n_algorithms = len(suggestions)
        
        if n_algorithms == 0:
            return {}
        
        # Garantizar mínimo
        guaranteed = self.min_budget * n_algorithms
        remaining = max(0, self.total_budget - guaranteed)
        
        # Calcular scores para asignación
        scores = []
        for sug in suggestions:
            # Score basado en confianza
            confidence_score = sug.confidence
            
            # Score basado en complejidad del espacio
            hp_space = HYPERPARAMETER_SPACES.get(sug.name)
            if hp_space:
                complexity_score = len(hp_space.parameters) / 10  # Normalizar
            else:
                complexity_score = 0.5
            
            # Score combinado
            score = (
                self.confidence_weight * confidence_score +
                self.complexity_weight * complexity_score
            )
            scores.append(score)
        
        # Normalizar scores
        total_score = sum(scores) + 1e-8
        proportions = [s / total_score for s in scores]
        
        # Asignar presupuesto
        allocation = {}
        for sug, prop in zip(suggestions, proportions):
            budget = self.min_budget + int(remaining * prop)
            allocation[sug.name] = budget
        
        # Ajustar si excede total
        total_allocated = sum(allocation.values())
        if total_allocated > self.total_budget:
            excess = total_allocated - self.total_budget
            # Reducir del algoritmo con más presupuesto
            max_alg = max(allocation, key=allocation.get)
            allocation[max_alg] -= excess
        
        return allocation
    
    def should_early_stop(
        self,
        history: List[float],
        patience: int = 10,
        min_improvement: float = 0.001
    ) -> bool:
        """
        Determina si parar temprano por convergencia.
        
        Returns:
            True si no hay mejora significativa en las últimas `patience` evaluaciones
        """
        if len(history) < patience:
            return False
        
        recent = history[-patience:]
        best_recent = max(recent)
        best_before = max(history[:-patience]) if len(history) > patience else 0
        
        improvement = best_recent - best_before
        return improvement < min_improvement


# =============================================================================
# MEJORA 3: Transfer de Hiperparámetros
# =============================================================================

class HyperparameterTransfer:
    """
    Transfiere hiperparámetros de tareas similares.
    
    Estrategia:
    - Ponderar configs por similitud de tarea
    - Interpolar entre configs similares
    - Mantener diversidad para exploración
    """
    
    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base
    
    def get_transfer_prior(
        self,
        dataset_meta: DatasetMetaFeatures,
        algorithm: str,
        n_configs: int = 10
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Genera prior de configuraciones basado en transfer.
        
        Returns:
            Lista de (config, weight) donde weight indica importancia
        """
        similar = self.kb.find_similar_configs(
            dataset_meta, algorithm, top_k=20
        )
        
        if not similar:
            return []
        
        # Ponderar por similitud y score
        weighted_configs = []
        
        for config, score, similarity in similar:
            # Weight combina similitud y score histórico
            weight = similarity * (0.5 + 0.5 * score)
            weighted_configs.append((config, weight))
        
        # Normalizar weights
        total_weight = sum(w for _, w in weighted_configs) + 1e-8
        weighted_configs = [(c, w/total_weight) for c, w in weighted_configs]
        
        # Tomar top-n
        weighted_configs.sort(key=lambda x: -x[1])
        return weighted_configs[:n_configs]
    
    def interpolate_configs(
        self,
        configs: List[Tuple[Dict[str, Any], float]],
        hp_space: HyperparameterSpace
    ) -> Dict[str, Any]:
        """
        Interpola configuraciones ponderadas.
        
        Returns:
            Configuración interpolada
        """
        if not configs:
            return {}
        
        # Convertir a vectores
        vectors = []
        weights = []
        
        for config, weight in configs:
            vec = hp_space.encode(config)
            vectors.append(vec)
            weights.append(weight)
        
        # Weighted average
        vectors = np.array(vectors)
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        interpolated = np.average(vectors, axis=0, weights=weights)
        
        # Decodificar
        return hp_space.decode(interpolated)


# =============================================================================
# Pipeline Principal
# =============================================================================

class FSBOPipeline:
    """
    Pipeline completo de optimización de hiperparámetros con FSBO.
    
    Integra:
    - Meta-features del dataset
    - Warm start inteligente
    - Ajuste dinámico de presupuesto
    - Transfer de hiperparámetros
    - Base de conocimiento
    
    Uso:
        pipeline = FSBOPipeline(total_budget=100)
        
        # Desde meta-learning
        suggestions = [
            AlgorithmSuggestion('random_forest', confidence=0.8),
            AlgorithmSuggestion('adaboost', confidence=0.6),
        ]
        
        result = pipeline.optimize(
            X_train, y_train, X_val, y_val,
            suggestions=suggestions,
            evaluation_fn=train_and_evaluate
        )
    """
    
    def __init__(
        self,
        total_budget: int = 100,
        min_budget_per_algorithm: int = 10,
        checkpoint_dir: Optional[str] = None,
        knowledge_base_path: Optional[str] = None,
        device: str = 'cpu',
        verbose: bool = True
    ):
        self.total_budget = total_budget
        self.min_budget = min_budget_per_algorithm
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.verbose = verbose
        
        # Componentes
        if knowledge_base_path is None:
            knowledge_base_path = Path(__file__).parent.parent / 'experiments' / 'knowledge_base.json'
        
        self.knowledge_base = KnowledgeBase(str(knowledge_base_path))
        self.warm_starter = IntelligentWarmStart(self.knowledge_base)
        self.budget_allocator = DynamicBudgetAllocator(
            total_budget=total_budget,
            min_budget_per_algorithm=min_budget_per_algorithm
        )
        self.hp_transfer = HyperparameterTransfer(self.knowledge_base)
    
    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        suggestions: List[AlgorithmSuggestion],
        evaluation_fn: Callable[[str, Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray], float],
        dataset_id: str = "unknown"
    ) -> PipelineResult:
        """
        Ejecuta el pipeline completo de optimización.
        
        Args:
            X_train, y_train: Datos de entrenamiento
            X_val, y_val: Datos de validación
            suggestions: Algoritmos sugeridos por meta-learning
            evaluation_fn: Función de evaluación
                          fn(algorithm, config, X_train, y_train, X_val, y_val) -> score
            dataset_id: Identificador del dataset
            
        Returns:
            PipelineResult con todos los resultados
        """
        start_time = datetime.now()
        
        if self.verbose:
            print("=" * 70)
            print("🚀 FSBO PIPELINE - Optimización de Hiperparámetros")
            print("=" * 70)
        
        # 1. Extraer meta-features
        if self.verbose:
            print("\n📊 Extrayendo meta-features del dataset...")
        
        meta_features = DatasetMetaFeatures.from_data(X_train, y_train, dataset_id)
        
        if self.verbose:
            print(f"   Samples: {meta_features.n_samples}")
            print(f"   Features: {meta_features.n_features}")
            print(f"   Classes: {meta_features.n_classes}")
            print(f"   Imbalance: {meta_features.class_imbalance:.3f}")
        
        # 2. Asignar presupuesto
        if self.verbose:
            print("\n💰 Asignando presupuesto por algoritmo...")
        
        budget_allocation = self.budget_allocator.allocate(suggestions)
        
        if self.verbose:
            for alg, budget in budget_allocation.items():
                conf = next((s.confidence for s in suggestions if s.name == alg), 0)
                print(f"   {alg}: {budget} evaluaciones (confianza={conf:.2f})")
        
        # 3. Optimizar cada algoritmo
        algorithm_results = {}
        total_evaluations = 0
        
        for suggestion in suggestions:
            algorithm = suggestion.name
            budget = budget_allocation.get(algorithm, self.min_budget)
            
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"🎯 Optimizando: {algorithm.upper()}")
                print(f"   Confianza meta-learning: {suggestion.confidence:.2f}")
                print(f"   Presupuesto: {budget} evaluaciones")
                print("=" * 70)
            
            result = self._optimize_algorithm(
                algorithm=algorithm,
                budget=budget,
                meta_features=meta_features,
                evaluation_fn=lambda cfg, alg=algorithm: evaluation_fn(
                    alg, cfg, X_train, y_train, X_val, y_val
                )
            )
            
            algorithm_results[algorithm] = result
            total_evaluations += result.n_evaluations
            
            # Guardar en base de conocimiento
            self.knowledge_base.add_entry(
                meta_features, algorithm, 
                result.best_config, result.best_score
            )
        
        # 4. Determinar mejor resultado
        best_algorithm = max(
            algorithm_results,
            key=lambda a: algorithm_results[a].best_score
        )
        best_result = algorithm_results[best_algorithm]
        
        # 5. Resultado final
        elapsed = (datetime.now() - start_time).total_seconds()
        
        pipeline_result = PipelineResult(
            dataset_id=dataset_id,
            meta_features=meta_features,
            algorithm_results=algorithm_results,
            best_algorithm=best_algorithm,
            best_config=best_result.best_config,
            best_score=best_result.best_score,
            total_evaluations=total_evaluations,
            total_time_seconds=elapsed,
            budget_allocation=budget_allocation
        )
        
        if self.verbose:
            self._print_summary(pipeline_result)
        
        return pipeline_result
    
    def _optimize_algorithm(
        self,
        algorithm: str,
        budget: int,
        meta_features: DatasetMetaFeatures,
        evaluation_fn: Callable[[Dict], float]
    ) -> OptimizationResult:
        """Optimiza un algoritmo individual."""
        
        # Cargar optimizador
        try:
            optimizer = FSBOOptimizer.from_pretrained(
                algorithm, 
                self.checkpoint_dir,
                self.device
            )
        except FileNotFoundError:
            if self.verbose:
                print(f"   ⚠️ No hay modelo pre-entrenado para {algorithm}")
                print(f"   Usando búsqueda aleatoria como fallback")
            return self._random_search_fallback(algorithm, evaluation_fn, budget)
        
        # Warm start inteligente (MEJORA 1)
        n_init = min(5, budget // 3)
        
        if self.verbose:
            print(f"\n   🔥 Warm Start Inteligente ({n_init} configs)...")
        
        initial_configs = self.warm_starter.get_initial_configs(
            meta_features, algorithm, optimizer, n_init
        )
        
        # Evaluar configs iniciales
        for i, config in enumerate(initial_configs):
            score = evaluation_fn(config)
            optimizer.observe(config, score)
            
            if self.verbose:
                print(f"      [{i+1}/{n_init}] Score: {score:.4f}")
        
        # Transfer prior (MEJORA 3)
        transfer_prior = self.hp_transfer.get_transfer_prior(
            meta_features, algorithm, n_configs=5
        )
        
        if transfer_prior and self.verbose:
            print(f"\n   📦 Transfer de {len(transfer_prior)} configs de tareas similares")
        
        # BO loop con early stopping (MEJORA 2)
        remaining = budget - n_init
        
        if self.verbose:
            print(f"\n   🔄 BO Loop ({remaining} iteraciones restantes)...")
        
        for i in range(remaining):
            # Early stopping check
            if self.budget_allocator.should_early_stop(optimizer.best_y_history):
                if self.verbose:
                    print(f"      ⏹️ Early stopping en iteración {i+1}")
                break
            
            # Suggest next config
            config = optimizer.suggest()
            
            # Ocasionalmente usar transfer prior
            if transfer_prior and np.random.random() < 0.2:
                # Sample from transfer prior
                weights = [w for _, w in transfer_prior]
                idx = np.random.choice(len(transfer_prior), p=weights)
                config = transfer_prior[idx][0]
            
            # Evaluate
            score = evaluation_fn(config)
            optimizer.observe(config, score)
            
            if self.verbose and (i + 1) % 5 == 0:
                best_cfg, best_score = optimizer.get_best()
                print(f"      [{n_init + i + 1}/{budget}] Best: {best_score:.4f}")
        
        result = optimizer.get_result()
        
        if self.verbose:
            print(f"\n   ✅ {algorithm} completado!")
            print(f"      Mejor score: {result.best_score:.4f}")
        
        return result
    
    def _random_search_fallback(
        self,
        algorithm: str,
        evaluation_fn: Callable[[Dict], float],
        budget: int
    ) -> OptimizationResult:
        """Fallback a búsqueda aleatoria."""
        hp_space = HYPERPARAMETER_SPACES.get(algorithm)
        
        if hp_space is None:
            hp_space = HyperparameterSpace(
                name=algorithm,
                parameters={f'hp_{i}': {'type': 'float', 'range': [0, 1]} 
                           for i in range(5)}
            )
        
        configs = []
        scores = []
        
        for i in range(budget):
            x = hp_space.sample_random(1)[0]
            config = hp_space.decode(x)
            score = evaluation_fn(config)
            configs.append(config)
            scores.append(score)
            
            if self.verbose and (i + 1) % 10 == 0:
                print(f"      [{i+1}/{budget}] Best: {max(scores):.4f}")
        
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
    
    def _print_summary(self, result: PipelineResult):
        """Imprime resumen de resultados."""
        print("\n" + "=" * 70)
        print("📋 RESUMEN DEL PIPELINE")
        print("=" * 70)
        
        print(f"\n📊 Dataset: {result.dataset_id}")
        print(f"   Samples: {result.meta_features.n_samples}")
        print(f"   Features: {result.meta_features.n_features}")
        
        print(f"\n⏱️ Tiempo total: {result.total_time_seconds:.1f} segundos")
        print(f"📈 Evaluaciones totales: {result.total_evaluations}")
        
        print(f"\n🏆 RESULTADOS POR ALGORITMO:")
        print("-" * 50)
        
        sorted_results = sorted(
            result.algorithm_results.items(),
            key=lambda x: -x[1].best_score
        )
        
        for i, (alg, res) in enumerate(sorted_results):
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
            print(f"{medal} {alg}:")
            print(f"      Score: {res.best_score:.4f}")
            print(f"      Evaluaciones: {res.n_evaluations}")
            # Mostrar config resumida
            config_str = str(res.best_config)[:50] + "..." if len(str(res.best_config)) > 50 else str(res.best_config)
            print(f"      Config: {config_str}")
        
        print(f"\n{'='*70}")
        print(f"🏆 MEJOR RESULTADO:")
        print(f"   Algoritmo: {result.best_algorithm}")
        print(f"   Score: {result.best_score:.4f}")
        print(f"   Configuración: {result.best_config}")
        print("=" * 70)


# =============================================================================
# Función de conveniencia para integración con Meta-Learning
# =============================================================================

def run_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    suggested_algorithms: List[Union[str, AlgorithmSuggestion]],
    evaluation_fn: Callable,
    total_budget: int = 100,
    dataset_id: str = "unknown",
    verbose: bool = True
) -> PipelineResult:
    """
    Función de conveniencia para ejecutar el pipeline completo.
    
    Esta es la función principal para integrar con meta-learning.
    
    Args:
        X_train, y_train: Datos de entrenamiento
        X_val, y_val: Datos de validación
        suggested_algorithms: Lista de algoritmos (str o AlgorithmSuggestion)
        evaluation_fn: Función fn(algorithm, config, X_train, y_train, X_val, y_val) -> score
        total_budget: Presupuesto total de evaluaciones
        dataset_id: Identificador del dataset
        verbose: Mostrar progreso
        
    Returns:
        PipelineResult
        
    Example:
        >>> from meta_learning import MetaLearner
        >>> 
        >>> # Meta-learning sugiere algoritmos
        >>> meta_learner = MetaLearner.load("model.pkl")
        >>> algorithms = meta_learner.suggest(X, y)
        >>> 
        >>> # Definir función de evaluación
        >>> def evaluate(algorithm, config, X_tr, y_tr, X_val, y_val):
        ...     model = get_model(algorithm, **config)
        ...     model.fit(X_tr, y_tr)
        ...     return model.score(X_val, y_val)
        >>> 
        >>> # Ejecutar pipeline
        >>> result = run_pipeline(
        ...     X_train, y_train, X_val, y_val,
        ...     suggested_algorithms=algorithms,
        ...     evaluation_fn=evaluate,
        ...     total_budget=100
        ... )
        >>> 
        >>> print(f"Mejor: {result.best_algorithm} con {result.best_score:.4f}")
    """
    # Convertir strings a AlgorithmSuggestion si es necesario
    suggestions = []
    for i, alg in enumerate(suggested_algorithms):
        if isinstance(alg, str):
            # Asignar confianza decreciente por orden
            confidence = 1.0 - (i * 0.1)
            suggestions.append(AlgorithmSuggestion(
                name=alg,
                confidence=max(0.5, confidence),
                expected_performance=0.0,
                priority=i
            ))
        else:
            suggestions.append(alg)
    
    # Crear y ejecutar pipeline
    pipeline = FSBOPipeline(
        total_budget=total_budget,
        verbose=verbose
    )
    
    return pipeline.optimize(
        X_train, y_train, X_val, y_val,
        suggestions=suggestions,
        evaluation_fn=evaluation_fn,
        dataset_id=dataset_id
    )


# =============================================================================
# CLI para testing
# =============================================================================

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    print("=" * 70)
    print("🧪 TEST: FSBO Pipeline Completo")
    print("=" * 70)
    
    # Generar dataset de prueba
    print("\n📊 Generando dataset sintético...")
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=3,
        random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"   Train: {X_train.shape}")
    print(f"   Val: {X_val.shape}")
    
    # Simular sugerencias del meta-learning
    print("\n🤖 Simulando sugerencias de meta-learning...")
    suggestions = [
        AlgorithmSuggestion('adaboost', confidence=0.85, expected_performance=0.0),
        AlgorithmSuggestion('random_forest', confidence=0.75, expected_performance=0.0),
    ]
    
    for s in suggestions:
        print(f"   {s.name}: confianza={s.confidence:.2f}")
    
    # Función de evaluación dummy
    def dummy_evaluate(algorithm, config, X_tr, y_tr, X_val, y_val):
        """Simula evaluación de un modelo."""
        # Simular score basado en algoritmo y config
        base_score = {'adaboost': 0.75, 'random_forest': 0.78}.get(algorithm, 0.70)
        
        # Añadir variación por config
        config_bonus = sum(
            0.02 if 0.3 < v < 0.7 else 0 
            for v in config.values() 
            if isinstance(v, (int, float))
        )
        
        # Ruido
        noise = np.random.normal(0, 0.02)
        
        return min(max(base_score + config_bonus + noise, 0.5), 1.0)
    
    # Ejecutar pipeline
    result = run_pipeline(
        X_train, y_train, X_val, y_val,
        suggested_algorithms=suggestions,
        evaluation_fn=dummy_evaluate,
        total_budget=40,
        dataset_id="test_synthetic",
        verbose=True
    )
    
    print("\n" + "=" * 70)
    print("✅ Test completado exitosamente!")
    print("=" * 70)

