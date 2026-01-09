"""
Meta-Learner basado en Transfer Learning.

Este módulo implementa un meta-learner que sugiere algoritmos basándose en:
- Meta-features del dataset
- Historial de optimizaciones (knowledge base)
- Similitud entre tareas

Autor: Proyecto MetaLearning
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AlgorithmSuggestion:
    """Sugerencia de algoritmo con confianza estimada."""
    name: str
    confidence: float
    expected_performance: float
    similar_tasks: int = 0


class TransferLearningMetaLearner:
    """
    Meta-Learner que sugiere algoritmos basándose en transfer learning.
    
    Funcionamiento:
    1. Recibe meta-features de un nuevo dataset
    2. Busca tareas similares en la knowledge base
    3. Rankea algoritmos por performance en tareas similares
    4. Devuelve top-k algoritmos con confianza
    
    Example:
        >>> from pipeline import KnowledgeBase, DatasetMetaFeatures
        >>> 
        >>> # Cargar knowledge base
        >>> kb = KnowledgeBase('experiments/knowledge_base.json')
        >>> 
        >>> # Crear meta-learner
        >>> meta_learner = TransferLearningMetaLearner(kb)
        >>> 
        >>> # Extraer meta-features de nuevo dataset
        >>> meta_features = DatasetMetaFeatures.from_data(X, y)
        >>> 
        >>> # Obtener sugerencias
        >>> suggestions = meta_learner.suggest_algorithms(meta_features, top_k=3)
        >>> 
        >>> for s in suggestions:
        ...     print(f"{s.name}: confianza={s.confidence:.2f}, "
        ...           f"performance esperado={s.expected_performance:.3f}")
    """
    
    def __init__(
        self,
        knowledge_base: 'KnowledgeBase',
        similarity_threshold: float = 0.5,
        min_similar_tasks: int = 1
    ):
        """
        Args:
            knowledge_base: Base de conocimiento con historial
            similarity_threshold: Umbral mínimo de similitud (0-1)
            min_similar_tasks: Mínimo de tareas similares requeridas
        """
        self.kb = knowledge_base
        self.similarity_threshold = similarity_threshold
        self.min_similar_tasks = min_similar_tasks
    
    def suggest_algorithms(
        self,
        dataset_meta: 'DatasetMetaFeatures',
        top_k: int = 3,
        max_similar: int = 10
    ) -> List[AlgorithmSuggestion]:
        """
        Sugiere los mejores algoritmos para un dataset.
        
        Args:
            dataset_meta: Meta-features del dataset objetivo
            top_k: Número de algoritmos a sugerir
            max_similar: Máximo de tareas similares a considerar por algoritmo
            
        Returns:
            Lista de AlgorithmSuggestion ordenadas por confianza
        """
        logger.info(f"Buscando algoritmos para dataset con meta-features: {dataset_meta.meta_vector[:3]}...")
        
        # 1. Encontrar todas las tareas similares
        similar_tasks = self._find_similar_tasks(dataset_meta, top_k=max_similar * 5)
        
        if not similar_tasks:
            logger.warning("No se encontraron tareas similares en la knowledge base")
            return self._get_default_suggestions(dataset_meta)
        
        logger.info(f"Encontradas {len(similar_tasks)} tareas similares")
        
        # 2. Agrupar por algoritmo y calcular estadísticas
        algorithm_stats = self._compute_algorithm_statistics(similar_tasks)
        
        # 3. Rankear algoritmos
        ranked = self._rank_algorithms(algorithm_stats)
        
        # 4. Crear sugerencias
        suggestions = []
        for algo_name, stats in ranked[:top_k]:
            suggestion = AlgorithmSuggestion(
                name=algo_name,
                confidence=stats['confidence'],
                expected_performance=stats['mean_score'],
                similar_tasks=stats['count']
            )
            suggestions.append(suggestion)
            
            logger.info(
                f"  • {algo_name}: confianza={suggestion.confidence:.3f}, "
                f"perf_esperado={suggestion.expected_performance:.3f}, "
                f"tareas_similares={suggestion.similar_tasks}"
            )
        
        return suggestions
    
    def _find_similar_tasks(
        self,
        dataset_meta: 'DatasetMetaFeatures',
        top_k: int = 50
    ) -> List[Tuple[Dict, float]]:
        """
        Encuentra tareas similares en la knowledge base.
        
        Returns:
            Lista de (entry, similarity) ordenadas por similitud
        """
        results = []
        
        for entry in self.kb.entries:
            # Calcular similitud coseno
            entry_vector = np.array(entry['meta_vector'])
            similarity = self._compute_similarity(
                dataset_meta.meta_vector,
                entry_vector
            )
            
            # Filtrar por umbral
            if similarity >= self.similarity_threshold:
                results.append((entry, similarity))
        
        # Ordenar por similitud descendente
        results.sort(key=lambda x: -x[1])
        
        return results[:top_k]
    
    def _compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calcula similitud entre dos vectores de meta-features.
        
        Usa similitud coseno + normalización por distancia euclidiana.
        """
        # Similitud coseno
        cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
        
        # Distancia euclidiana normalizada
        dist = np.linalg.norm(vec1 - vec2)
        dist_sim = 1 / (1 + dist)
        
        # Combinar ambas métricas
        combined_sim = 0.7 * cos_sim + 0.3 * dist_sim
        
        return max(0, combined_sim)  # Asegurar [0, 1]
    
    def _compute_algorithm_statistics(
        self,
        similar_tasks: List[Tuple[Dict, float]]
    ) -> Dict[str, Dict]:
        """
        Calcula estadísticas por algoritmo en tareas similares.
        
        Returns:
            {algorithm: {'scores': [...], 'similarities': [...], ...}}
        """
        stats = {}
        
        for entry, similarity in similar_tasks:
            algo = entry['algorithm']
            score = entry['best_score']
            
            if algo not in stats:
                stats[algo] = {
                    'scores': [],
                    'similarities': [],
                    'configs': []
                }
            
            stats[algo]['scores'].append(score)
            stats[algo]['similarities'].append(similarity)
            stats[algo]['configs'].append(entry['best_config'])
        
        # Calcular estadísticas agregadas
        for algo, data in stats.items():
            scores = np.array(data['scores'])
            sims = np.array(data['similarities'])
            
            # Performance esperado (weighted by similarity)
            weights = sims / sims.sum()
            mean_score = np.average(scores, weights=weights)
            std_score = np.std(scores)
            
            # Confianza basada en:
            # - Número de tareas similares
            # - Similitud promedio
            # - Consistencia (bajo std)
            count = len(scores)
            mean_sim = np.mean(sims)
            consistency = 1 / (1 + std_score)
            
            confidence = (
                0.4 * min(count / 10, 1.0) +      # Más tareas → más confianza
                0.4 * mean_sim +                   # Mayor similitud → más confianza
                0.2 * consistency                  # Menor varianza → más confianza
            )
            
            stats[algo]['mean_score'] = mean_score
            stats[algo]['std_score'] = std_score
            stats[algo]['mean_similarity'] = mean_sim
            stats[algo]['confidence'] = confidence
            stats[algo]['count'] = count
        
        return stats
    
    def _rank_algorithms(
        self,
        algorithm_stats: Dict[str, Dict]
    ) -> List[Tuple[str, Dict]]:
        """
        Rankea algoritmos por score esperado y confianza.
        
        Returns:
            Lista de (algorithm, stats) ordenada
        """
        # Filtrar algoritmos con pocas tareas similares
        filtered = {
            algo: stats
            for algo, stats in algorithm_stats.items()
            if stats['count'] >= self.min_similar_tasks
        }
        
        if not filtered:
            logger.warning("Ningún algoritmo tiene suficientes tareas similares")
            return list(algorithm_stats.items())
        
        # Ordenar por score combinado: performance + confianza
        ranked = sorted(
            filtered.items(),
            key=lambda x: (
                0.6 * x[1]['mean_score'] +        # 60% performance
                0.4 * x[1]['confidence']           # 40% confianza
            ),
            reverse=True
        )
        
        return ranked
    
    def _get_default_suggestions(
        self,
        dataset_meta: Optional['DatasetMetaFeatures'] = None
    ) -> List[AlgorithmSuggestion]:
        """
        Devuelve sugerencias por defecto cuando no hay knowledge base.
        
        Si se proveen meta-features, ajusta las sugerencias basándose en
        características del dataset (heurísticas simples).
        """
        logger.info("Usando sugerencias por defecto (sin knowledge base)")
        
        # Sugerencias base
        defaults = [
            ('random_forest', 0.7, 0.80),
            ('gradient_boosting', 0.65, 0.78),
            ('adaboost', 0.6, 0.75),
        ]
        
        # Si tenemos meta-features, ajustar heurísticamente
        if dataset_meta is not None:
            defaults = self._adjust_defaults_by_metafeatures(defaults, dataset_meta)
        
        return [
            AlgorithmSuggestion(
                name=name,
                confidence=conf,
                expected_performance=perf,
                similar_tasks=0
            )
            for name, conf, perf in defaults
        ]
    
    def _adjust_defaults_by_metafeatures(
        self,
        defaults: List[Tuple[str, float, float]],
        dataset_meta: 'DatasetMetaFeatures'
    ) -> List[Tuple[str, float, float]]:
        """
        Ajusta sugerencias por defecto usando heurísticas basadas en meta-features.
        
        Heurísticas simples:
        - Dataset grande → priorizar algoritmos eficientes
        - Pocas features → tree-based funciona bien
        - Muchas features → gradient boosting mejor
        - Clases desbalanceadas → random forest más robusto
        """
        import math
        
        # Extraer info de meta-features (aproximado)
        # meta_vector ~ [log(n_samples), log(n_features), log(n_classes), ...]
        log_n_samples = dataset_meta.meta_vector[0]
        log_n_features = dataset_meta.meta_vector[1] if len(dataset_meta.meta_vector) > 1 else 2.0
        
        # Calcular características
        n_samples_approx = math.exp(log_n_samples)
        n_features_approx = math.exp(log_n_features)
        
        is_large = n_samples_approx > 10000
        is_high_dim = n_features_approx > 50
        is_small = n_samples_approx < 500
        
        # Ajustar confianzas
        adjusted = []
        for algo, conf, perf in defaults:
            new_conf = conf
            new_perf = perf
            
            if algo == 'random_forest':
                if is_large:
                    new_conf += 0.05  # RF escala bien
                if is_high_dim:
                    new_conf -= 0.05  # RF puede overfittear en alta dim
                if is_small:
                    new_conf += 0.05  # RF robusto en datasets pequeños
                    
            elif algo == 'gradient_boosting':
                if is_high_dim:
                    new_conf += 0.05  # GB maneja alta dimensionalidad
                if is_small:
                    new_conf -= 0.05  # GB puede overfittear
                if is_large:
                    new_conf -= 0.05  # GB más lento
                    
            elif algo == 'adaboost':
                if is_small:
                    new_conf += 0.05  # AdaBoost bien en pequeño
                if is_high_dim:
                    new_conf -= 0.05  # AdaBoost débil en alta dim
            
            # Clip confianzas
            new_conf = max(0.3, min(0.9, new_conf))
            
            adjusted.append((algo, new_conf, new_perf))
        
        # Reordenar por confianza
        adjusted.sort(key=lambda x: -x[1])
        
        logger.info(f"Ajustadas sugerencias por meta-features:")
        logger.info(f"  • Samples≈{int(n_samples_approx)}, Features≈{int(n_features_approx)}")
        for algo, conf, _ in adjusted:
            logger.info(f"  • {algo}: {conf:.3f}")
        
        return adjusted
    
    def get_warm_start_configs(
        self,
        dataset_meta: 'DatasetMetaFeatures',
        algorithm: str,
        n_configs: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Obtiene configuraciones para warm start de un algoritmo.
        
        Args:
            dataset_meta: Meta-features del dataset
            algorithm: Nombre del algoritmo
            n_configs: Número de configuraciones a devolver
            
        Returns:
            Lista de configuraciones de tareas similares
        """
        # Buscar tareas similares para este algoritmo específico
        similar_tasks = self._find_similar_tasks(dataset_meta, top_k=50)
        
        # Filtrar por algoritmo
        algo_tasks = [
            (entry, sim)
            for entry, sim in similar_tasks
            if entry['algorithm'] == algorithm
        ]
        
        if not algo_tasks:
            logger.warning(f"No se encontraron tareas similares para {algorithm}")
            return []
        
        # Ordenar por similitud y tomar top-k
        algo_tasks.sort(key=lambda x: -x[1])
        
        configs = [entry['best_config'] for entry, _ in algo_tasks[:n_configs]]
        
        logger.info(f"Encontradas {len(configs)} configs de warm start para {algorithm}")
        
        return configs


# =============================================================================
# Funciones de utilidad
# =============================================================================

def analyze_knowledge_base(kb: 'KnowledgeBase') -> Dict[str, Any]:
    """
    Analiza la knowledge base y devuelve estadísticas.
    
    Returns:
        Diccionario con estadísticas de la KB
    """
    if not kb.entries:
        return {'total_entries': 0}
    
    algorithms = {}
    for entry in kb.entries:
        algo = entry['algorithm']
        if algo not in algorithms:
            algorithms[algo] = {'count': 0, 'scores': []}
        algorithms[algo]['count'] += 1
        algorithms[algo]['scores'].append(entry['best_score'])
    
    # Calcular estadísticas por algoritmo
    for algo, data in algorithms.items():
        scores = np.array(data['scores'])
        data['mean_score'] = float(np.mean(scores))
        data['std_score'] = float(np.std(scores))
        data['min_score'] = float(np.min(scores))
        data['max_score'] = float(np.max(scores))
    
    return {
        'total_entries': len(kb.entries),
        'algorithms': algorithms,
        'datasets': len(set(e['dataset_id'] for e in kb.entries))
    }

