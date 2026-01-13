"""
Pipeline de Producción para Meta-Learning + FSBO.

Este pipeline está diseñado para uso en producción con:
- Datasets reales
- Meta-learner basado en transfer learning
- Knowledge base persistente
- Evaluación real de modelos

Autor: Proyecto MetaLearning
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass
import logging
from datetime import datetime

# Imports locales
from pipeline import (
    KnowledgeBase,
    DatasetMetaFeatures,
    PipelineResult
)
from meta_learner import TransferLearningMetaLearner, AlgorithmSuggestion
from fsbo_optimizer import FSBOOptimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProductionConfig:
    """Configuración del pipeline de producción."""
    # Paths
    knowledge_base_path: str = "experiments/knowledge_base.json"
    checkpoints_dir: str = "experiments/checkpoints"
    results_dir: str = "experiments/results"
    
    # Budget
    total_budget: int = 100
    min_budget_per_algorithm: int = 10
    max_algorithms: int = 5
    
    # Meta-learner
    similarity_threshold: float = 0.5
    min_similar_tasks: int = 2
    
    # Warm start
    n_warm_start: int = 5
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_threshold: float = 0.001
    
    # Device
    device: str = 'cpu'
    
    # Verbose
    verbose: bool = True


class ProductionPipeline:
    """
    Pipeline de producción para optimización de hiperparámetros con transfer learning.
    
    Flujo:
    1. Recibe dataset real (X, y)
    2. Extrae meta-features
    3. Meta-learner sugiere algoritmos (basado en knowledge base)
    4. FSBO optimiza hiperparámetros de cada algoritmo
    5. Guarda resultados en knowledge base
    6. Devuelve mejor configuración
    
    Example:
        ```python
        from sklearn.datasets import fetch_openml
        from production_pipeline import ProductionPipeline, ProductionConfig
        
        # Cargar dataset real
        data = fetch_openml('credit-g', version=1, parser='auto')
        X, y = data.data, data.target
        
        # Crear pipeline
        config = ProductionConfig(
            total_budget=100,
            max_algorithms=3,
            verbose=True
        )
        pipeline = ProductionPipeline(config)
        
        # Ejecutar
        result = pipeline.run(
            X, y,
            dataset_id='credit-g',
            test_size=0.2
        )
        
        # Usar mejor modelo
        print(f"Mejor algoritmo: {result.best_algorithm}")
        print(f"Mejor config: {result.best_config}")
        print(f"Score: {result.best_score:.4f}")
        ```
    """
    
    def __init__(self, config: ProductionConfig = None):
        """
        Args:
            config: Configuración del pipeline (usa defaults si None)
        """
        self.config = config or ProductionConfig()
        
        # Crear directorios
        Path(self.config.results_dir).mkdir(parents=True, exist_ok=True)
        
        # Cargar knowledge base
        self.kb = KnowledgeBase(self.config.knowledge_base_path)
        
        # Crear meta-learner
        self.meta_learner = TransferLearningMetaLearner(
            self.kb,
            similarity_threshold=self.config.similarity_threshold,
            min_similar_tasks=self.config.min_similar_tasks
        )
        
        if self.config.verbose:
            kb_stats = self._analyze_kb()
            print("\n" + "=" * 70)
            print("🚀 PRODUCTION PIPELINE INICIALIZADO")
            print("=" * 70)
            print(f"\n📦 Knowledge Base:")
            print(f"   • Entradas totales: {kb_stats['total_entries']}")
            print(f"   • Datasets únicos: {kb_stats.get('datasets', 0)}")
            print(f"   • Algoritmos: {', '.join(kb_stats.get('algorithms', {}).keys())}")
    
    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dataset_id: str,
        test_size: float = 0.2,
        evaluation_fn: Optional[Callable] = None,
        save_to_kb: bool = True
    ) -> PipelineResult:
        """
        Ejecuta el pipeline completo de optimización.
        
        Args:
            X: Features del dataset
            y: Labels
            dataset_id: Identificador único del dataset
            test_size: Proporción para validación
            evaluation_fn: Función custom de evaluación (opcional)
                          def eval(algorithm, config, X_tr, y_tr, X_val, y_val) -> float
            save_to_kb: Si guardar resultados en knowledge base
            
        Returns:
            PipelineResult con mejor configuración encontrada
        """
        start_time = datetime.now()
        
        if self.config.verbose:
            print("\n" + "=" * 70)
            print(f"📊 DATASET: {dataset_id}")
            print("=" * 70)
            print(f"   • Samples: {len(X)}")
            print(f"   • Features: {X.shape[1]}")
            print(f"   • Classes: {len(np.unique(y))}")
        
        # 1. Split datos
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        if self.config.verbose:
            print(f"   • Train: {X_train.shape}")
            print(f"   • Validation: {X_val.shape}")
        
        # 2. Extraer meta-features
        if self.config.verbose:
            print("\n🔍 Extrayendo meta-features...")
        
        meta_features = DatasetMetaFeatures.from_data(
            X_train, y_train,
            dataset_id=dataset_id
        )
        
        if self.config.verbose:
            print(f"   ✓ Meta-vector: {meta_features.meta_vector[:4]}...")
        
        # 3. Meta-learner sugiere algoritmos
        if self.config.verbose:
            print("\n🤖 Meta-Learner sugiriendo algoritmos...")
        
        suggestions = self.meta_learner.suggest_algorithms(
            meta_features,
            top_k=self.config.max_algorithms
        )
        
        if self.config.verbose:
            print(f"\n   Algoritmos sugeridos:")
            for i, s in enumerate(suggestions, 1):
                print(f"   {i}. {s.name:<20} confianza={s.confidence:.3f}  "
                      f"perf_esperado={s.expected_performance:.3f}  "
                      f"tareas_similares={s.similar_tasks}")
        
        # 4. Asignar presupuesto
        budgets = self._allocate_budget(suggestions)
        
        if self.config.verbose:
            print(f"\n💰 Presupuesto asignado:")
            for algo, budget in budgets.items():
                print(f"   • {algo}: {budget} evaluaciones")
        
        # 5. Función de evaluación
        if evaluation_fn is None:
            evaluation_fn = self._default_evaluation
        
        # 6. Optimizar cada algoritmo
        results = {}
        total_evals = 0
        
        for suggestion in suggestions:
            algo = suggestion.name
            budget = budgets[algo]
            
            if self.config.verbose:
                print("\n" + "=" * 70)
                print(f"🎯 OPTIMIZANDO: {algo.upper()}")
                print(f"   Presupuesto: {budget} evaluaciones")
                print("=" * 70)
            
            result = self._optimize_algorithm(
                algo, budget, suggestion,
                meta_features, evaluation_fn,
                X_train, y_train, X_val, y_val
            )
            
            results[algo] = result
            total_evals += result.n_evaluations
            
            if self.config.verbose:
                print(f"\n   ✅ Completado!")
                print(f"   Mejor score: {result.best_score:.4f}")
                print(f"   Evaluaciones: {result.n_evaluations}")
        
        # 7. Seleccionar mejor algoritmo
        best_algo = max(results.keys(), key=lambda a: results[a].best_score)
        best_result = results[best_algo]
        
        # 8. Guardar en knowledge base
        if save_to_kb:
            self.kb.add_entry(
                meta_features,
                best_algo,
                best_result.best_config,
                best_result.best_score
            )
            
            if self.config.verbose:
                print(f"\n💾 Guardado en knowledge base")
        
        # 9. Crear resultado final
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        pipeline_result = PipelineResult(
            best_algorithm=best_algo,
            best_config=best_result.best_config,
            best_score=best_result.best_score,
            total_evaluations=total_evals,
            algorithm_results={k: v.__dict__ for k, v in results.items()},
            meta_features=meta_features.meta_vector.tolist(),
            dataset_id=dataset_id,
            duration=duration,
            timestamp=end_time.isoformat()
        )
        
        # 10. Guardar resultados
        self._save_results(pipeline_result)
        
        # 11. Mostrar resumen
        if self.config.verbose:
            self._print_summary(pipeline_result, results)
        
        return pipeline_result
    
    def _optimize_algorithm(
        self,
        algorithm: str,
        budget: int,
        suggestion: AlgorithmSuggestion,
        meta_features: DatasetMetaFeatures,
        evaluation_fn: Callable,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ):
        """Optimiza un algoritmo específico."""
        from dataclasses import dataclass
        
        @dataclass
        class OptResult:
            best_config: Dict
            best_score: float
            n_evaluations: int
            history: List[float]
        
        # Cargar optimizador FSBO
        try:
            optimizer = FSBOOptimizer.from_pretrained(
                algorithm,
                checkpoint_dir=self.config.checkpoints_dir,
                device=self.config.device
            )
        except FileNotFoundError:
            logger.warning(f"No checkpoint para {algorithm}, usando random search")
            return self._random_search(
                algorithm, budget, evaluation_fn,
                X_train, y_train, X_val, y_val
            )
        
        # Warm start con configs de tareas similares
        warm_configs = self.meta_learner.get_warm_start_configs(
            meta_features, algorithm, n_configs=self.config.n_warm_start
        )
        
        n_init = min(self.config.n_warm_start, budget // 3)
        
        if warm_configs and self.config.verbose:
            print(f"\n   🔥 Warm Start con {len(warm_configs)} configs de tareas similares")
        
        # Usar warm configs o generar aleatorias
        if warm_configs:
            initial_configs = warm_configs[:n_init]
        else:
            initial_configs = optimizer.suggest_initial(n_init)
        
        # Evaluar configs iniciales
        for i, config in enumerate(initial_configs):
            score = evaluation_fn(algorithm, config, X_train, y_train, X_val, y_val)
            optimizer.observe(config, score)
            
            if self.config.verbose and (i + 1) % 2 == 0:
                print(f"      [{i+1}/{n_init}] Score: {score:.4f}")
        
        # BO loop con early stopping
        remaining = budget - n_init
        best_scores = []
        patience_counter = 0
        
        if self.config.verbose:
            print(f"\n   🔄 BO Loop ({remaining} iteraciones)...")
        
        for i in range(remaining):
            config = optimizer.suggest()
            score = evaluation_fn(algorithm, config, X_train, y_train, X_val, y_val)
            optimizer.observe(config, score)
            
            best_config, best_score = optimizer.get_best()
            best_scores.append(best_score)
            
            # Early stopping
            if len(best_scores) > self.config.early_stopping_patience:
                recent = best_scores[-self.config.early_stopping_patience:]
                improvement = max(recent) - min(recent)
                
                if improvement < self.config.early_stopping_threshold:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        if self.config.verbose:
                            print(f"\n   ⏹️ Early stopping en iteración {i+1}")
                        break
                else:
                    patience_counter = 0
            
            if self.config.verbose and (i + 1) % 10 == 0:
                print(f"      [{n_init + i + 1}/{budget}] Best: {best_score:.4f}")
        
        result = optimizer.get_result()
        
        return OptResult(
            best_config=result.best_config,
            best_score=result.best_score,
            n_evaluations=result.n_evaluations,
            history=result.history
        )
    
    def _allocate_budget(
        self,
        suggestions: List[AlgorithmSuggestion]
    ) -> Dict[str, int]:
        """Asigna presupuesto basado en confianza."""
        if not suggestions:
            return {}
        
        # Normalizar confianzas
        total_conf = sum(s.confidence for s in suggestions)
        
        budgets = {}
        remaining = self.config.total_budget
        
        for s in suggestions[:-1]:
            budget = max(
                self.config.min_budget_per_algorithm,
                int((s.confidence / total_conf) * self.config.total_budget)
            )
            budgets[s.name] = budget
            remaining -= budget
        
        # Último algoritmo recibe lo que queda
        budgets[suggestions[-1].name] = max(
            self.config.min_budget_per_algorithm,
            remaining
        )
        
        return budgets
    
    def _default_evaluation(
        self,
        algorithm: str,
        config: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> float:
        """Evaluación por defecto usando sklearn."""
        from sklearn.ensemble import (
            AdaBoostClassifier,
            RandomForestClassifier,
            GradientBoostingClassifier
        )
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier
        
        # Mapear algoritmos a clases
        models = {
            'adaboost': AdaBoostClassifier,
            'random_forest': RandomForestClassifier,
            'gradient_boosting': GradientBoostingClassifier,
            'libsvm_svc': SVC,
            'svm': SVC,
        }
        
        if algorithm not in models:
            logger.warning(f"Algoritmo {algorithm} no soportado, retornando 0.5")
            return 0.5
        
        try:
            # Limpiar config (remover prefijos si existen)
            clean_config = {
                k.split('__')[-1]: v
                for k, v in config.items()
                if not k.startswith('imputation')
            }
            
            # Crear y entrenar modelo
            model = models[algorithm](**clean_config)
            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            
            return float(score)
            
        except Exception as e:
            logger.error(f"Error evaluando {algorithm}: {e}")
            return 0.0
    
    def _random_search(self, algorithm, budget, eval_fn, X_tr, y_tr, X_val, y_val):
        """Fallback a random search."""
        from fsbo_optimizer import HYPERPARAMETER_SPACES
        from dataclasses import dataclass
        
        @dataclass
        class OptResult:
            best_config: Dict
            best_score: float
            n_evaluations: int
            history: List[float]
        
        hp_space = HYPERPARAMETER_SPACES.get(algorithm)
        if not hp_space:
            return OptResult({}, 0.0, 0, [])
        
        configs = []
        scores = []
        
        for _ in range(budget):
            x = hp_space.sample_random(1)[0]
            config = hp_space.decode(x)
            score = eval_fn(algorithm, config, X_tr, y_tr, X_val, y_val)
            configs.append(config)
            scores.append(score)
        
        best_idx = np.argmax(scores)
        history = [max(scores[:i+1]) for i in range(len(scores))]
        
        return OptResult(
            best_config=configs[best_idx],
            best_score=scores[best_idx],
            n_evaluations=budget,
            history=history
        )
    
    def _save_results(self, result: PipelineResult):
        """Guarda resultados a disco."""
        import json
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"production_{result.dataset_id}_{timestamp}.json"
        filepath = Path(self.config.results_dir) / filename
        
        with open(filepath, 'w') as f:
            json.dump(result.__dict__, f, indent=2, default=str)
        
        logger.info(f"Resultados guardados en: {filepath}")
    
    def _print_summary(self, result: PipelineResult, results: Dict):
        """Imprime resumen de resultados."""
        print("\n" + "=" * 70)
        print("📋 RESUMEN FINAL")
        print("=" * 70)
        
        print(f"\n📊 Dataset: {result.dataset_id}")
        print(f"⏱️  Tiempo total: {result.duration:.1f} segundos")
        print(f"📈 Evaluaciones totales: {result.total_evaluations}")
        
        print(f"\n🏆 MEJOR RESULTADO:")
        print(f"   • Algoritmo: {result.best_algorithm}")
        print(f"   • Score: {result.best_score:.4f}")
        print(f"   • Configuración:")
        for k, v in list(result.best_config.items())[:5]:
            print(f"      - {k}: {v}")
        if len(result.best_config) > 5:
            print(f"      ... ({len(result.best_config) - 5} más)")
        
        print(f"\n📊 RESULTADOS POR ALGORITMO:")
        sorted_algos = sorted(
            results.items(),
            key=lambda x: x[1].best_score,
            reverse=True
        )
        
        for i, (algo, res) in enumerate(sorted_algos, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            print(f"   {medal} {algo:<20} Score: {res.best_score:.4f}  "
                  f"Evals: {res.n_evaluations}")
        
        print("\n" + "=" * 70)
    
    def _analyze_kb(self) -> Dict:
        """Analiza knowledge base."""
        from meta_learner import analyze_knowledge_base
        return analyze_knowledge_base(self.kb)


# =============================================================================
# Función helper para uso rápido
# =============================================================================

def optimize_dataset(
    X: np.ndarray,
    y: np.ndarray,
    dataset_id: str,
    total_budget: int = 100,
    max_algorithms: int = 3,
    verbose: bool = True
) -> PipelineResult:
    """
    Función de alto nivel para optimizar un dataset rápidamente.
    
    Example:
        >>> from sklearn.datasets import load_iris
        >>> from production_pipeline import optimize_dataset
        >>> 
        >>> X, y = load_iris(return_X_y=True)
        >>> result = optimize_dataset(X, y, 'iris', total_budget=50)
        >>> 
        >>> print(f"Mejor: {result.best_algorithm} con score {result.best_score:.3f}")
    """
    config = ProductionConfig(
        total_budget=total_budget,
        max_algorithms=max_algorithms,
        verbose=verbose
    )
    
    pipeline = ProductionPipeline(config)
    return pipeline.run(X, y, dataset_id)


# =============================================================================
# Main para testing
# =============================================================================

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    
    print("🧪 Testing Production Pipeline\n")
    
    # Generar dataset de prueba
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=15,
        n_classes=3,
        random_state=42
    )
    
    # Ejecutar pipeline
    result = optimize_dataset(
        X, y,
        dataset_id="test_production",
        total_budget=60,
        max_algorithms=3,
        verbose=True
    )
    
    print("\n✅ Test completado!")

