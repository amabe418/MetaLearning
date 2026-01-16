"""
Integración MetaFeatX + FSBO: Pipeline en Dos Fases

Este módulo implementa la integración completa:
- Fase 1: MetaFeatX (entrenado) sugiere algoritmos basándose en el nuevo dataset
- Fase 2: FSBO optimiza hiperparámetros de cada algoritmo sugerido

Flujo:
1. Nuevo dataset (X, y) → extraer meta-features básicas
2. Entrenar/usar MetaFeatX con datos históricos
3. MetaFeatX predice representaciones y encuentra vecinos
4. Calcular scores y rankear algoritmos
5. Para cada algoritmo sugerido → FSBO optimiza hiperparámetros
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Callable
from dataclasses import dataclass
import logging

# Agregar paths para imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "transfer-learning" / "scripts"))

# Imports de MetaFeatX
from model.metafeatx import MetaFeatX
from experiment.utils import load_basic_features, load_target_features, load_bootstrap_features
from experiment.reprs import get_model_representations
from ranking.score import calculate_general_score, mean_top_k_accuracy_per_task, calculate_individual_score

# Imports de FSBO
try:
    from fsbo_optimizer import optimize_algorithms, OptimizationResult
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent / "transfer-learning" / "scripts"))
    from fsbo_optimizer import optimize_algorithms, OptimizationResult

# Imports para extracción de meta-features
try:
    from pymfe.mfe import MFE
    PYMFE_AVAILABLE = True
except ImportError:
    PYMFE_AVAILABLE = False
    logging.warning("pymfe no disponible. No se pueden extraer meta-features completas.")

# Imports para configuración
try:
    from omegaconf import OmegaConf
    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False

from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class IntegrationResult:
    """Resultado de la integración completa."""
    suggested_algorithms: List[str]
    algorithm_scores: Dict[str, float]
    optimization_results: Dict[str, OptimizationResult]
    best_algorithm: str
    best_config: Dict[str, Any]
    best_score: float
    total_evaluations: int


class MetaFeatXFSBOIntegration:
    """
    Pipeline de integración MetaFeatX + FSBO en dos fases.
    
    Usa MetaFeatX entrenado para sugerir algoritmos y luego FSBO
    para optimizar hiperparámetros de cada algoritmo sugerido.
    
    Args:
        data_path: Ruta a los datos históricos (default: './data')
        conf_path: Ruta a configuración (default: './conf')
        checkpoint_dir: Directorio de checkpoints FSBO
        verbose: Mostrar progreso
    """
    
    def __init__(
        self,
        data_path: str = "./data",
        conf_path: str = "./conf",
        checkpoint_dir: Optional[str] = None,
        verbose: bool = True
    ):
        self.data_path = Path(data_path)
        self.conf_path = Path(conf_path)
        self.checkpoint_dir = checkpoint_dir or str(Path(__file__).parent / "transfer-learning" / "experiments" / "checkpoints")
        self.verbose = verbose
        
        # Algoritmos disponibles
        self.available_algorithms = ["adaboost", "random_forest", "libsvm_svc"]
        
        # Cargar configuración
        self._load_config()
    
    def _load_config(self):
        """Carga configuración necesaria."""
        try:
            if OMEGACONF_AVAILABLE:
                cfg_main = OmegaConf.load(self.conf_path / "config.yaml")
                cfg_metafeature = OmegaConf.load(self.conf_path / "metafeature" / "metafeatx.yaml")
                cfg_task = OmegaConf.load(self.conf_path / "task" / "task1.yaml")
                
                self.metafeature_config = cfg_metafeature
                self.task_config = cfg_task
                self.seed = cfg_main.get("seed", 42)
            else:
                # Valores por defecto
                self.metafeature_config = type('obj', (object,), {
                    'name': 'metafeatx',
                    'basic_columns': 'one_itemset,two_itemset,leaves,leaves_branch,leaves_corrob,leaves_homo,leaves_per_class,nodes,nodes_per_attr,nodes_per_inst,nodes_per_level,nodes_repeated,tree_depth,tree_imbalance,tree_shape,var_importance,best_node,elite_nn,ch,int,nre,pb,sc,sil,vdb,vdu,cohesiveness,conceptvar,impconceptvar,wg_dist,cls_coef,density,f1,f1v,f2,f3,f4,hubs,l1,l2,l3,lsc,n1,n2,n3,n4,t1,t2,t3,t4,attr_conc,attr_ent,class_conc,joint_ent,mut_inf,ns_ratio,can_cor,cor,cov,eigenvalues,g_mean,gravity,h_mean,iq_range,lh_trace,mad,max,mean,median,min,nr_cor_attr,nr_disc,nr_norm,nr_outliers,p_trace,range,roy_root,sd,sd_ratio,sparsity,t_mean,var,w_lambda,attr_to_inst,cat_to_num,freq_class,inst_to_attr,nr_attr,nr_bin,PCASkewnessFirstPC,PCAKurtosisFirstPC,PCAFractionOfComponentsFor95PercentVariance,Landmark1NN,LandmarkRandomNodeLearner,LandmarkDecisionNodeLearner,LandmarkDecisionTree,LandmarkNaiveBayes,LandmarkLDA,ClassEntropy,SkewnessSTD,SkewnessMean,SkewnessMax,SkewnessMin,KurtosisSTD,KurtosisMean,KurtosisMax,KurtosisMin,SymbolsSum,SymbolsSTD,SymbolsMean,SymbolsMax,SymbolsMin,ClassProbabilitySTD,ClassProbabilityMean,ClassProbabilityMax,ClassProbabilityMin,InverseDatasetRatio,DatasetRatio,RatioNominalToNumerical,RatioNumericalToNominal,NumberOfCategoricalFeatures,NumberOfNumericFeatures,NumberOfMissingValues,NumberOfFeaturesWithMissingValues,NumberOfInstancesWithMissingValues,NumberOfFeatures,NumberOfClasses,NumberOfInstances,LogInverseDatasetRatio,LogDatasetRatio,PercentageOfMissingValues,PercentageOfFeaturesWithMissingValues,PercentageOfInstancesWithMissingValues,LogNumberOfFeatures,LogNumberOfInstances'
                })()
                self.task_config = type('obj', (object,), {'ndcg': 10})()
                self.seed = 42
                
        except Exception as e:
            logger.warning(f"Error cargando configuración: {e}")
            # Valores por defecto
            self.metafeature_config = type('obj', (object,), {
                'name': 'metafeatx',
                'basic_columns': 'one_itemset,two_itemset,leaves,leaves_branch,leaves_corrob,leaves_homo,leaves_per_class,nodes,nodes_per_attr,nodes_per_inst,nodes_per_level,nodes_repeated,tree_depth,tree_imbalance,tree_shape,var_importance,best_node,elite_nn,ch,int,nre,pb,sc,sil,vdb,vdu,cohesiveness,conceptvar,impconceptvar,wg_dist,cls_coef,density,f1,f1v,f2,f3,f4,hubs,l1,l2,l3,lsc,n1,n2,n3,n4,t1,t2,t3,t4,attr_conc,attr_ent,class_conc,joint_ent,mut_inf,ns_ratio,can_cor,cor,cov,eigenvalues,g_mean,gravity,h_mean,iq_range,lh_trace,mad,max,mean,median,min,nr_cor_attr,nr_disc,nr_norm,nr_outliers,p_trace,range,roy_root,sd,sd_ratio,sparsity,t_mean,var,w_lambda,attr_to_inst,cat_to_num,freq_class,inst_to_attr,nr_attr,nr_bin,PCASkewnessFirstPC,PCAKurtosisFirstPC,PCAFractionOfComponentsFor95PercentVariance,Landmark1NN,LandmarkRandomNodeLearner,LandmarkDecisionNodeLearner,LandmarkDecisionTree,LandmarkNaiveBayes,LandmarkLDA,ClassEntropy,SkewnessSTD,SkewnessMean,SkewnessMax,SkewnessMin,KurtosisSTD,KurtosisMean,KurtosisMax,KurtosisMin,SymbolsSum,SymbolsSTD,SymbolsMean,SymbolsMax,SymbolsMin,ClassProbabilitySTD,ClassProbabilityMean,ClassProbabilityMax,ClassProbabilityMin,InverseDatasetRatio,DatasetRatio,RatioNominalToNumerical,RatioNumericalToNominal,NumberOfCategoricalFeatures,NumberOfNumericFeatures,NumberOfMissingValues,NumberOfFeaturesWithMissingValues,NumberOfInstancesWithMissingValues,NumberOfFeatures,NumberOfClasses,NumberOfInstances,LogInverseDatasetRatio,LogDatasetRatio,PercentageOfMissingValues,PercentageOfFeaturesWithMissingValues,PercentageOfInstancesWithMissingValues,LogNumberOfFeatures,LogNumberOfInstances'
            })()
            self.task_config = type('obj', (object,), {'ndcg': 10})()
            self.seed = 42
    
    def extract_basic_meta_features(self, X: np.ndarray, y: np.ndarray, task_id: int = -1) -> pd.DataFrame:
        """
        Extrae meta-features básicas del nuevo dataset usando pymfe.
        
        Args:
            X: Features del dataset
            y: Target del dataset
            task_id: ID de la tarea (usar -1 para nuevo dataset)
            
        Returns:
            DataFrame con meta-features básicas
        """
        if not PYMFE_AVAILABLE:
            raise ImportError("pymfe no está disponible. Instala con: pip install pymfe")
        
        if self.verbose:
            print(f"\n📊 Extrayendo meta-features básicas...")
        
        # Extraer meta-features usando pymfe
        mfe = MFE(groups=["all"])
        mfe.fit(X, y)
        ft_names, ft_values = mfe.extract()
        
        df = pd.DataFrame([ft_values], columns=ft_names)
        df["task_id"] = task_id
        
        # Manejar arrays como t1
        if "t1" in df.columns:
            t1_values = df.loc[0, "t1"]
            if isinstance(t1_values, (list, np.ndarray)):
                t1_array = np.array(t1_values, dtype=float)
                df["t1.mean"] = t1_array.mean()
                df["t1.sd"] = t1_array.std()
        
        # Seleccionar solo las columnas que necesitamos
        required_cols = ["task_id"] + self.metafeature_config.basic_columns.split(",")
        available_cols = [col for col in required_cols if col in df.columns]
        
        # Rellenar columnas faltantes con 0
        for col in required_cols:
            if col not in df.columns and col != "task_id":
                df[col] = 0.0
        
        df = df[required_cols]
        df = df.fillna(0)
        
        if self.verbose:
            print(f"  ✓ Meta-features extraídas: {len(required_cols)} características")
        
        return df
    
    def suggest_algorithms_with_metafeatx(
        self,
        X: np.ndarray,
        y: np.ndarray,
        top_k: int = 3,
        k_neighbors: int = 10,
        k_top_configs: int = 20
    ) -> Tuple[List[str], Dict[str, float], Dict[str, Dict[int, List[int]]]]:
        """
        Fase 1: Sugiere algoritmos usando MetaFeatX entrenado.
        
        Este método:
        1. Extrae meta-features básicas del nuevo dataset
        2. Carga datos históricos
        3. Entrena MetaFeatX (o usa uno pre-entrenado)
        4. Predice representaciones y encuentra vecinos
        5. Calcula scores por algoritmo
        6. Retorna top-k algoritmos
        
        Args:
            X: Features del dataset
            y: Target del dataset
            top_k: Número de algoritmos a sugerir
            k_neighbors: Número de vecinos a considerar
            k_top_configs: Número de top configuraciones a promediar
            
        Returns:
            Tuple de (lista de algoritmos sugeridos, diccionario de scores, all_neighbors)
            - all_neighbors: {algorithm: {task_id: [neighbor_ids]}}
        """
        if self.verbose:
            print("\n" + "=" * 70)
            print("📊 FASE 1: Sugerencia de Algoritmos con MetaFeatX")
            print("=" * 70)
        
        # 1. Extraer meta-features básicas del nuevo dataset
        new_meta_features = self.extract_basic_meta_features(X, y, task_id=-1)
        
        # 2. Cargar datos históricos
        if self.verbose:
            print(f"\n📂 Cargando datos históricos...")
        
        # Cargar basic representations históricas
        basic_reprs_historical = load_basic_features(
            metafeature=self.metafeature_config,
            path=str(self.data_path)
        )
        
        if basic_reprs_historical.empty:
            raise ValueError(f"No se encontraron datos históricos en {self.data_path}")
        
        if self.verbose:
            print(f"  ✓ {len(basic_reprs_historical)} tareas históricas cargadas")
        
        # Combinar nuevo dataset con históricos
        basic_reprs_all = pd.concat([basic_reprs_historical, new_meta_features], ignore_index=True)
        
        # 3. Para cada algoritmo, entrenar MetaFeatX y encontrar vecinos
        all_neighbors = {}
        
        # Normalizar meta-features básicas (necesario para usar con MetaFeatX entrenado)
        cols_to_normalize = [col for col in basic_reprs_historical.columns if col != "task_id"]
        scaler = StandardScaler()
        basic_reprs_historical_normalized = basic_reprs_historical.copy()
        basic_reprs_historical_normalized[cols_to_normalize] = scaler.fit_transform(
            basic_reprs_historical[cols_to_normalize]
        )
        
        # Normalizar también el nuevo dataset con el mismo scaler
        new_meta_features_normalized = new_meta_features.copy()
        available_cols = [col for col in cols_to_normalize if col in new_meta_features.columns]
        new_meta_features_normalized[available_cols] = scaler.transform(
            new_meta_features[available_cols]
        )
        
        for algorithm in self.available_algorithms:
            if self.verbose:
                print(f"\n🔍 Procesando algoritmo: {algorithm}")
            
            try:
                # Cargar target representations para este algoritmo
                class PipelineConfig:
                    def __init__(self, name):
                        self.name = name
                
                pipeline_cfg = PipelineConfig(algorithm)
                target_reprs = load_target_features(pipeline=pipeline_cfg, path=str(self.data_path))
                
                # Obtener lista de task_ids históricos
                list_ids = sorted(list(target_reprs["task_id"].unique()))
                
                # Filtrar basic_reprs para incluir solo históricos
                basic_reprs_train = basic_reprs_historical_normalized[
                    basic_reprs_historical_normalized.task_id.isin(list_ids)
                ]
                
                # Entrenar MetaFeatX con datos históricos
                if self.verbose:
                    print(f"  Entrenando MetaFeatX con {len(list_ids)} tareas históricas...")
                
                meta_model = MetaFeatX(
                    alpha=0.5,
                    lambda_reg=1e-3,
                    learning_rate=0.01,
                    early_stopping_patience=20,
                    early_stopping_criterion_ndcg=self.task_config.ndcg,
                    verbose=False,
                    seed=self.seed
                )
                
                meta_model.train(
                    basic_reprs=basic_reprs_train,
                    target_reprs=target_reprs,
                    column_id="task_id"
                )
                
                # Predecir representaciones para datos históricos y nuevo dataset
                historical_predicted = meta_model.predict(basic_reprs=basic_reprs_train)
                new_predicted = meta_model.predict(basic_reprs=new_meta_features_normalized)
                
                # Calcular distancias entre nuevo dataset y todos los históricos
                distances = pairwise_distances(new_predicted, historical_predicted, metric='euclidean')[0]
                
                # Obtener k vecinos más cercanos
                top_k_indices = np.argsort(distances)[:k_neighbors]
                neighbors = [list_ids[i] for i in top_k_indices]
                
                all_neighbors[algorithm] = {-1: neighbors}  # -1 es el ID del nuevo dataset
                
                if self.verbose:
                    print(f"  ✓ Encontrados {len(neighbors)} vecinos más cercanos")
                    print(f"    Vecinos: {neighbors[:5]}...")
                
            except Exception as e:
                logger.warning(f"Error procesando {algorithm}: {e}")
                import traceback
                traceback.print_exc()
                # Si falla, usar vecinos vacíos
                all_neighbors[algorithm] = {-1: []}
        
        # 4. Calcular scores por algoritmo usando calculate_general_score
        if self.verbose:
            print(f"\n📊 Calculando scores por algoritmo...")
        
        scores_list = calculate_general_score(
            all_neighbors, 
            task_id=-1, 
            k=k_top_configs,
            data_path=str(self.data_path)
        )
        
        # Convertir a diccionario
        algorithm_scores = {item["pipeline"]: item["score"] for item in scores_list}
        
        # Seleccionar top-k algoritmos
        sorted_algorithms = sorted(
            algorithm_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        suggested_algorithms = [algo for algo, score in sorted_algorithms[:top_k]]
        scores_dict = {algo: score for algo, score in sorted_algorithms[:top_k]}
        
        if self.verbose:
            print(f"\n✓ Algoritmos sugeridos (top {top_k}):")
            for i, (algo, score) in enumerate(sorted_algorithms[:top_k], 1):
                print(f"  {i}. {algo:<20} score: {score:.4f}")
        
        # Retornar también los vecinos para usar como warm start
        return suggested_algorithms, scores_dict, all_neighbors
    
    def get_warm_start_configs_from_neighbors(
        self,
        algorithm: str,
        neighbors: List[int],
        k_top_configs_per_neighbor: int = 3,
        max_total_configs: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Extrae configuraciones del top-k de los vecinos encontrados por MetaFeatX.
        
        Estas configuraciones se usarán como warm start para FSBO.
        Las configuraciones vienen directamente de los archivos CSV que tienen
        las mejores configuraciones de cada vecino.
        
        Args:
            algorithm: Nombre del algoritmo
            neighbors: Lista de task_ids de vecinos (encontrados por MetaFeatX)
            k_top_configs_per_neighbor: Número de top configuraciones a extraer por vecino
            max_total_configs: Máximo total de configuraciones a retornar
            
        Returns:
            Lista de configuraciones de hiperparámetros (diccionarios)
        """
        warm_configs = []
        
        try:
            # Cargar archivo con configuraciones del top-k
            target_csv = self.data_path / "top_raw_target_representation" / f"{algorithm}_target_representation.csv"
            
            if not target_csv.exists():
                if self.verbose:
                    print(f"   ⚠️  No se encuentra {target_csv}, usando warm start por defecto")
                return []
            
            df = pd.read_csv(target_csv)
            
            # Para cada vecino, extraer top-k configuraciones
            for neighbor_id in neighbors:
                if len(warm_configs) >= max_total_configs:
                    break
                    
                neighbor_rows = df[df["task_id"] == neighbor_id]
                
                if neighbor_rows.empty:
                    continue
                
                # Ordenar por predictive_accuracy y tomar top-k
                top_configs = neighbor_rows.nlargest(k_top_configs_per_neighbor, "predictive_accuracy")
                
                # Convertir cada fila a diccionario de configuración
                for _, row in top_configs.iterrows():
                    if len(warm_configs) >= max_total_configs:
                        break
                        
                    config = {}
                    
                    # Excluir columnas que no son hiperparámetros
                    exclude_cols = ["task_id", "predictive_accuracy"]
                    
                    for col in df.columns:
                        if col not in exclude_cols:
                            # Mantener nombre original (puede tener prefijos como "classifier__")
                            # FSBO sabe cómo manejar estos nombres
                            config[col] = row[col]
                    
                    if config:  # Solo añadir si tiene hiperparámetros
                        warm_configs.append(config)
            
            if self.verbose and warm_configs:
                print(f"   ✓ Extraídas {len(warm_configs)} configs de warm start de {len(neighbors)} vecinos")
            
        except Exception as e:
            logger.warning(f"Error extrayendo warm start configs para {algorithm}: {e}")
            import traceback
            traceback.print_exc()
            return []
        
        return warm_configs
    
    def optimize_with_fsbo(
        self,
        algorithms: List[str],
        evaluation_fn: Callable[[str, Dict[str, Any]], float],
        budget_per_algorithm: int = 30,
        n_init: int = 5,
        warm_start_configs: Optional[Dict[str, List[Dict[str, Any]]]] = None
    ) -> Dict[str, OptimizationResult]:
        """
        Fase 2: Optimiza hiperparámetros usando FSBO.
        
        Args:
            algorithms: Lista de algoritmos a optimizar
            evaluation_fn: Función que recibe (algorithm, config) y retorna score
            budget_per_algorithm: Evaluaciones por algoritmo
            n_init: Configuraciones iniciales (si no hay warm_start_configs)
            warm_start_configs: Dict[algorithm, List[configs]] - Configuraciones de warm start
                              desde MetaFeatX (vecinos)
            
        Returns:
            Diccionario con resultados de optimización por algoritmo
        """
        if self.verbose:
            print("\n" + "=" * 70)
            print("🚀 FASE 2: Optimización de Hiperparámetros (FSBO)")
            print("=" * 70)
        
        # Adaptar función de evaluación para formato de optimize_algorithms
        def adapted_eval_fn(algorithm: str, config: Dict[str, Any]) -> float:
            return evaluation_fn(algorithm, config)
        
        # Optimizar cada algoritmo individualmente para poder pasar warm start
        results = {}
        
        for algorithm in algorithms:
            if self.verbose:
                print(f"\n🎯 Optimizando: {algorithm}")
            
            # Obtener warm start configs para este algoritmo
            warm_configs = None
            if warm_start_configs and algorithm in warm_start_configs:
                warm_configs = warm_start_configs[algorithm]
                if self.verbose and warm_configs:
                    print(f"   🔥 Warm Start: {len(warm_configs)} configs de MetaFeatX")
            
            # Optimizar algoritmo individual
            result = self._optimize_algorithm_with_warm_start(
                algorithm=algorithm,
                evaluation_fn=lambda cfg: adapted_eval_fn(algorithm, cfg),
                budget=budget_per_algorithm,
                n_init=n_init,
                warm_start_configs=warm_configs
            )
            
            results[algorithm] = result
        
        return results
    
    def _optimize_algorithm_with_warm_start(
        self,
        algorithm: str,
        evaluation_fn: Callable[[Dict[str, Any]], float],
        budget: int = 30,
        n_init: int = 5,
        warm_start_configs: Optional[List[Dict[str, Any]]] = None
    ) -> 'OptimizationResult':
        """
        Optimiza un algoritmo individual con warm start opcional.
        
        Si warm_start_configs está disponible, usa esas configuraciones.
        Si no, usa suggest_initial del optimizador.
        """
        from fsbo_optimizer import FSBOOptimizer, OptimizationResult
        
        # Cargar optimizador
        try:
            optimizer = FSBOOptimizer.from_pretrained(
                algorithm, 
                checkpoint_dir=self.checkpoint_dir
            )
        except FileNotFoundError:
            if self.verbose:
                print(f"   ⚠️  No hay modelo pre-entrenado para {algorithm}")
                print(f"   Usando búsqueda aleatoria como fallback")
            # Fallback a búsqueda aleatoria
            from fsbo_optimizer import HYPERPARAMETER_SPACES, HyperparameterSpace
            import numpy as np
            
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
        
        # Determinar configuraciones iniciales
        if warm_start_configs and len(warm_start_configs) > 0:
            # Usar configuraciones de MetaFeatX (vecinos)
            initial_configs = warm_start_configs[:min(n_init, len(warm_start_configs))]
            
            if self.verbose:
                print(f"   🔥 Warm Start con {len(initial_configs)} configs de MetaFeatX")
        else:
            # Usar suggest_initial del optimizador
            initial_configs = optimizer.suggest_initial(n_init)
            
            if self.verbose:
                print(f"   🔄 Warm Start con {len(initial_configs)} configs del modelo FSBO")
        
        # Evaluar configs iniciales
        for i, config in enumerate(initial_configs):
            try:
                score = evaluation_fn(config)
                optimizer.observe(config, score)
                
                if self.verbose:
                    print(f"      [{i+1}/{len(initial_configs)}] Score: {score:.4f}")
            except Exception as e:
                logger.warning(f"Error evaluando config inicial {i+1}: {e}")
                # Continuar con la siguiente
        
        # BO loop
        remaining = budget - len(initial_configs)
        
        if self.verbose:
            print(f"\n   🔄 BO Loop ({remaining} iteraciones restantes)...")
        
        for i in range(remaining):
            config = optimizer.suggest()
            score = evaluation_fn(config)
            optimizer.observe(config, score)
            
            if self.verbose and (i + 1) % 5 == 0:
                best_cfg, best_score = optimizer.get_best()
                print(f"      [{len(initial_configs) + i + 1}/{budget}] Best: {best_score:.4f}")
        
        result = optimizer.get_result()
        
        if self.verbose:
            print(f"\n   ✅ {algorithm} completado!")
            print(f"      Mejor score: {result.best_score:.4f}")
        
        return result
    
    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        evaluation_fn: Callable[[str, Dict[str, Any]], float],
        top_k_algorithms: int = 3,
        budget_per_algorithm: int = 30,
        n_init: int = 5,
        k_neighbors: int = 10
    ) -> IntegrationResult:
        """
        Ejecuta el pipeline completo de integración.
        
        Args:
            X: Features del dataset
            y: Target del dataset
            evaluation_fn: Función de evaluación (algorithm, config) -> score
            top_k_algorithms: Número de algoritmos a sugerir
            budget_per_algorithm: Evaluaciones por algoritmo
            n_init: Configuraciones iniciales
            k_neighbors: Vecinos a considerar para sugerencias
            
        Returns:
            IntegrationResult con todos los resultados
        """
        # Fase 1: Sugerir algoritmos con MetaFeatX
        suggested_algorithms, algorithm_scores, all_neighbors = self.suggest_algorithms_with_metafeatx(
            X, y,
            top_k=top_k_algorithms,
            k_neighbors=k_neighbors
        )
        
        # Extraer configuraciones de warm start de los vecinos encontrados por MetaFeatX
        warm_start_configs = {}
        for algorithm in suggested_algorithms:
            # Obtener vecinos para este algoritmo
            neighbors = all_neighbors.get(algorithm, {}).get(-1, [])
            
            if neighbors:
                # Extraer configuraciones del top-k de los vecinos
                warm_configs = self.get_warm_start_configs_from_neighbors(
                    algorithm=algorithm,
                    neighbors=neighbors,
                    k_top_configs_per_neighbor=3,  # Top 3 configs por vecino
                    max_total_configs=n_init  # Máximo igual a n_init
                )
                
                if warm_configs:
                    warm_start_configs[algorithm] = warm_configs
                    if self.verbose:
                        print(f"\n   📦 Warm Start para {algorithm}: {len(warm_configs)} configs de vecinos MetaFeatX")
        
        # Fase 2: Optimizar con FSBO (usando warm start de MetaFeatX)
        optimization_results = self.optimize_with_fsbo(
            algorithms=suggested_algorithms,
            evaluation_fn=evaluation_fn,
            budget_per_algorithm=budget_per_algorithm,
            n_init=n_init,
            warm_start_configs=warm_start_configs if warm_start_configs else None
        )
        
        # Encontrar mejor algoritmo
        best_algorithm = max(
            optimization_results.keys(),
            key=lambda a: optimization_results[a].best_score
        )
        best_result = optimization_results[best_algorithm]
        
        # Calcular total de evaluaciones
        total_evaluations = sum(r.n_evaluations for r in optimization_results.values())
        
        # Crear resultado
        result = IntegrationResult(
            suggested_algorithms=suggested_algorithms,
            algorithm_scores=algorithm_scores,
            optimization_results=optimization_results,
            best_algorithm=best_algorithm,
            best_config=best_result.best_config,
            best_score=best_result.best_score,
            total_evaluations=total_evaluations
        )
        
        # Mostrar resumen
        if self.verbose:
            print("\n" + "=" * 70)
            print("📋 RESUMEN FINAL")
            print("=" * 70)
            print(f"\n✓ Mejor algoritmo: {best_algorithm}")
            print(f"✓ Mejor score: {best_result.best_score:.4f}")
            print(f"✓ Total evaluaciones: {total_evaluations}")
            print(f"\n✓ Resultados por algoritmo:")
            for algo in suggested_algorithms:
                res = optimization_results[algo]
                print(f"  • {algo:<20} score: {res.best_score:.4f} ({res.n_evaluations} evals)")
        
        return result


# =============================================================================
# Función de conveniencia
# =============================================================================

def integrate_metafeatx_fsbo(
    X: np.ndarray,
    y: np.ndarray,
    evaluation_fn: Callable[[str, Dict[str, Any]], float],
    data_path: str = "./data",
    conf_path: str = "./conf",
    checkpoint_dir: Optional[str] = None,
    top_k_algorithms: int = 3,
    budget_per_algorithm: int = 30,
    verbose: bool = True
) -> IntegrationResult:
    """
    Función de conveniencia para ejecutar la integración completa.
    
    Args:
        X: Features del dataset
        y: Target del dataset
        evaluation_fn: Función de evaluación (algorithm, config) -> score
        data_path: Ruta a datos históricos
        conf_path: Ruta a configuración
        checkpoint_dir: Directorio de checkpoints FSBO
        top_k_algorithms: Número de algoritmos a sugerir
        budget_per_algorithm: Evaluaciones por algoritmo
        verbose: Mostrar progreso
        
    Returns:
        IntegrationResult con todos los resultados
    """
    integration = MetaFeatXFSBOIntegration(
        data_path=data_path,
        conf_path=conf_path,
        checkpoint_dir=checkpoint_dir,
        verbose=verbose
    )
    
    return integration.run(
        X=X,
        y=y,
        evaluation_fn=evaluation_fn,
        top_k_algorithms=top_k_algorithms,
        budget_per_algorithm=budget_per_algorithm
    )


if __name__ == "__main__":
    # Ejemplo de uso
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    
    print("Ejemplo de uso de integración MetaFeatX + FSBO\n")
    
    # Cargar dataset
    X, y = load_iris(return_X_y=True)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Función de evaluación
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
            return 0.5
        
        try:
            model = models[algorithm](**clean_config)
            model.fit(X_train, y_train)
            return float(model.score(X_val, y_val))
        except Exception as e:
            print(f"Error en {algorithm}: {e}")
            return 0.0
    
    # Ejecutar integración
    result = integrate_metafeatx_fsbo(
        X=X,
        y=y,
        evaluation_fn=evaluate,
        top_k_algorithms=3,
        budget_per_algorithm=20,
        verbose=True
    )
    
    print(f"\n✓ Integración completada!")
    print(f"  Mejor algoritmo: {result.best_algorithm}")
    print(f"  Mejor score: {result.best_score:.4f}")
