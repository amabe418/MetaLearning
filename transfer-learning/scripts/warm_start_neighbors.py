"""
Warm-Start basado en K-Vecinos usando MetaFeatX.

Este módulo implementa la estrategia de warm-start basada en k-vecinos:
- Usa el modelo MetaFeatX para encontrar datasets similares al nuevo
- Extrae las mejores configuraciones históricas de esos datasets vecinos
- Usa esas configuraciones como punto de partida para la optimización

Esta estrategia es DIFERENTE al warm-start de FSBO (transfer learning):
- FSBO warm-start: usa el GP meta-entrenado para predecir configs prometedoras
- MetaFeatX warm-start: usa configs REALES de datasets similares

Ambas estrategias pueden compararse experimentalmente.

Uso:
    from warm_start_neighbors import MetaFeatXWarmStart
    
    # Estrategia MetaFeatX (k-vecinos)
    warm_starter = MetaFeatXWarmStart.from_data(
        algorithm='random_forest',
        data_path='./data'
    )
    
    # Dado un nuevo dataset, obtener configs de vecinos
    initial_configs = warm_starter.suggest_initial(
        task_id=123,
        n=5
    )

Autor: Proyecto académico MetaLearning
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

# Agregar path del proyecto
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


# =============================================================================
# Configuración
# =============================================================================

# Mapeo de nombre de algoritmo en CSV a nombre interno
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

# Columnas del pipeline (one-hot en los CSVs)
PIPELINE_COLUMNS_PREFIX = [
    'Imputer Strategy_',
    'Categorical Strategy_',
    'Feature Selection_',
    'Scaler_',
]

# Columnas de metadata a excluir
METADATA_COLUMNS = ['Dataset', 'Fold', 'Fold Accuracy', 'Training Time', 'Testing Time', 'random_state']


# =============================================================================
# Estructuras de datos
# =============================================================================

@dataclass
class WarmStartConfig:
    """Configuración sugerida para warm-start."""
    pipeline: Dict[str, str]
    classifier: Dict[str, Any]
    source_dataset: str
    source_accuracy: float
    distance_to_new: float


# =============================================================================
# MetaFeatXWarmStart
# =============================================================================

class MetaFeatXWarmStart:
    """
    Estrategia de Warm-Start basada en K-Vecinos (MetaFeatX).
    
    Esta estrategia:
    1. Proyecta datasets a un espacio de embedding usando MetaFeatX
    2. Dado un nuevo dataset, encuentra los k datasets más similares
    3. Extrae las mejores configuraciones históricas de esos vecinos
    4. Usa esas configuraciones como warm-start
    
    DIFERENCIA CON FSBO WARM-START:
    - FSBO: predice configs prometedoras usando el GP meta-entrenado
    - MetaFeatX: usa configs REALES que ya funcionaron en datasets similares
    
    Interfaz compatible con FSBOPipelineOptimizer para facilitar comparación.
    """
    
    def __init__(
        self,
        algorithm: str,
        basic_representations: pd.DataFrame,
        historical_data: pd.DataFrame,
        task_id_mapping: Dict[str, int],
        metafeatx_model: Optional[Any] = None,
        embedding_matrix: Optional[np.ndarray] = None,
    ):
        """
        Args:
            algorithm: Nombre del algoritmo (ej: 'random_forest')
            basic_representations: DataFrame con metafeatures básicas (task_id + features)
            historical_data: DataFrame con configuraciones históricas (pipes/combined/*.csv)
            task_id_mapping: Mapeo Dataset (string) → task_id (int)
            metafeatx_model: Modelo MetaFeatX entrenado (opcional)
            embedding_matrix: Matriz de embeddings pre-calculada (opcional)
        """
        self.algorithm = algorithm
        self.basic_representations = basic_representations
        self.historical_data = historical_data
        self.task_id_mapping = task_id_mapping
        self.reverse_mapping = {v: k for k, v in task_id_mapping.items()}
        
        self.metafeatx_model = metafeatx_model
        self.embedding_matrix = embedding_matrix
        
        # Lista de task_ids disponibles
        self.available_task_ids = list(basic_representations['task_id'].unique())
        
        # Si no hay embedding, usar representaciones básicas directamente
        if self.embedding_matrix is None:
            self._compute_simple_embedding()
    
    def _compute_simple_embedding(self):
        """
        Calcula embedding simple (sin MetaFeatX) usando las representaciones básicas.
        
        Esto es un fallback cuando no hay modelo MetaFeatX disponible.
        Usa PCA o simplemente las features normalizadas.
        """
        # Obtener features (todas menos task_id)
        feature_cols = [c for c in self.basic_representations.columns if c != 'task_id']
        
        # Agregar por task_id (promedio de bootstraps)
        aggregated = self.basic_representations.groupby('task_id')[feature_cols].mean()
        
        # Normalizar
        scaler = StandardScaler()
        self.embedding_matrix = scaler.fit_transform(aggregated.values)
        self.embedding_task_ids = list(aggregated.index)
        
        logger.info(f"Embedding simple calculado: {self.embedding_matrix.shape}")
    
    @classmethod
    def from_data(
        cls,
        algorithm: str,
        data_path: str = './data',
        pipes_path: str = './pipes/combined',
        use_metafeatx: bool = False,
    ) -> 'MetaFeatXWarmStart':
        """
        Crea un NeighborWarmStarter desde los datos del proyecto.
        
        Args:
            algorithm: Nombre del algoritmo
            data_path: Ruta a la carpeta data/
            pipes_path: Ruta a pipes/combined/
            use_metafeatx: Si usar el modelo MetaFeatX (requiere entrenarlo)
            
        Returns:
            NeighborWarmStarter configurado
        """
        data_path = Path(data_path)
        pipes_path = Path(pipes_path)
        
        # 1. Cargar metafeatures básicas
        basic_repr_path = data_path / 'basic_representations.csv'
        if not basic_repr_path.exists():
            raise FileNotFoundError(f"No encontrado: {basic_repr_path}")
        
        basic_representations = pd.read_csv(basic_repr_path)
        logger.info(f"Cargadas {len(basic_representations)} filas de basic_representations")
        
        # 2. Cargar datos históricos del algoritmo
        csv_name = ALGORITHM_CSV_MAPPING.get(algorithm, algorithm)
        historical_path = pipes_path / f'{csv_name}_combined.csv'
        
        if not historical_path.exists():
            raise FileNotFoundError(f"No encontrado: {historical_path}")
        
        historical_data = pd.read_csv(historical_path)
        logger.info(f"Cargadas {len(historical_data)} configuraciones históricas")
        
        # 3. Crear mapeo Dataset → task_id
        # Intentar encontrar correspondencia
        task_id_mapping = cls._create_task_mapping(
            basic_representations, 
            historical_data
        )
        
        logger.info(f"Mapeo creado: {len(task_id_mapping)} datasets")
        
        # 4. Crear instancia
        return cls(
            algorithm=algorithm,
            basic_representations=basic_representations,
            historical_data=historical_data,
            task_id_mapping=task_id_mapping,
        )
    
    @staticmethod
    def _create_task_mapping(
        basic_repr: pd.DataFrame, 
        historical: pd.DataFrame
    ) -> Dict[str, int]:
        """
        Crea mapeo entre nombres de Dataset (string) y task_id (int).
        
        Intenta hacer match entre los datasets de basic_representations
        y los de historical_data.
        """
        # task_ids en basic_representations
        available_task_ids = set(basic_repr['task_id'].unique())
        
        # Datasets en historical_data
        if 'Dataset' in historical.columns:
            datasets = historical['Dataset'].unique()
        else:
            datasets = []
        
        mapping = {}
        
        # Intentar extraer task_id del nombre del dataset
        # Formato común: "nombre_taskid" o "nombre_taskid_algo"
        for dataset in datasets:
            parts = str(dataset).split('_')
            
            # Buscar un número que sea task_id
            for part in reversed(parts):
                try:
                    potential_id = int(part)
                    if potential_id in available_task_ids:
                        mapping[dataset] = potential_id
                        break
                except ValueError:
                    continue
        
        # Si no encontramos mapeos, crear uno simple basado en orden
        if not mapping:
            for i, dataset in enumerate(sorted(datasets)):
                if i < len(available_task_ids):
                    mapping[dataset] = sorted(available_task_ids)[i]
        
        return mapping
    
    def get_neighbors(
        self,
        task_id: int,
        k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Encuentra los k datasets más similares a un task_id dado.
        
        Args:
            task_id: ID del dataset de consulta
            k: Número de vecinos
            
        Returns:
            Lista de (task_id, distancia) ordenada por distancia
        """
        if task_id not in self.embedding_task_ids:
            raise ValueError(f"task_id {task_id} no encontrado en embeddings")
        
        # Índice del task_id
        query_idx = self.embedding_task_ids.index(task_id)
        query_embedding = self.embedding_matrix[query_idx:query_idx+1]
        
        # Calcular distancias
        distances = pairwise_distances(
            query_embedding, 
            self.embedding_matrix, 
            metric='euclidean'
        )[0]
        
        # Ordenar y excluir el propio dataset
        sorted_indices = np.argsort(distances)
        
        neighbors = []
        for idx in sorted_indices:
            if self.embedding_task_ids[idx] != task_id:
                neighbors.append((
                    self.embedding_task_ids[idx],
                    distances[idx]
                ))
            if len(neighbors) >= k:
                break
        
        return neighbors
    
    def get_best_configs_for_dataset(
        self,
        dataset_name: str,
        n: int = 3
    ) -> List[Dict]:
        """
        Obtiene las n mejores configuraciones para un dataset.
        
        Args:
            dataset_name: Nombre del dataset en historical_data
            n: Número de configuraciones a retornar
            
        Returns:
            Lista de diccionarios con configuración
        """
        # Filtrar por dataset
        df = self.historical_data[self.historical_data['Dataset'] == dataset_name]
        
        if df.empty:
            return []
        
        # Ordenar por accuracy
        df = df.nlargest(n, 'Fold Accuracy')
        
        configs = []
        for _, row in df.iterrows():
            config = self._row_to_config(row)
            config['_source_dataset'] = dataset_name
            config['_source_accuracy'] = row['Fold Accuracy']
            configs.append(config)
        
        return configs
    
    def _row_to_config(self, row: pd.Series) -> Dict:
        """Convierte una fila del CSV a diccionario de configuración."""
        config = {'pipeline': {}, 'classifier': {}}
        
        # Extraer pipeline
        for col in row.index:
            # Imputer Strategy
            if col.startswith('Imputer Strategy_') and row[col] == 1:
                config['pipeline']['imputer_strategy'] = col.replace('Imputer Strategy_', '')
            # Categorical Strategy
            elif col.startswith('Categorical Strategy_') and row[col] == 1:
                config['pipeline']['categorical_strategy'] = col.replace('Categorical Strategy_', '')
            # Feature Selection
            elif col.startswith('Feature Selection_') and row[col] == 1:
                config['pipeline']['feature_selection'] = col.replace('Feature Selection_', '')
            # Scaler
            elif col.startswith('Scaler_') and row[col] == 1:
                config['pipeline']['scaler'] = col.replace('Scaler_', '')
        
        # Extraer hiperparámetros del clasificador
        for col in row.index:
            if col not in METADATA_COLUMNS and not any(col.startswith(p) for p in PIPELINE_COLUMNS_PREFIX):
                value = row[col]
                # Convertir tipos
                if pd.notna(value):
                    if isinstance(value, (np.integer, int)):
                        value = int(value)
                    elif isinstance(value, (np.floating, float)):
                        # Mantener como float si tiene decimales
                        if value == int(value):
                            value = int(value)
                        else:
                            value = float(value)
                    elif isinstance(value, str):
                        # Convertir strings booleanos
                        if value.lower() == 'true':
                            value = True
                        elif value.lower() == 'false':
                            value = False
                    
                    config['classifier'][col] = value
        
        return config
    
    def get_warm_start_configs(
        self,
        task_id: int,
        k_neighbors: int = 5,
        configs_per_neighbor: int = 2,
        deduplicate: bool = True
    ) -> List[WarmStartConfig]:
        """
        Obtiene configuraciones para warm-start basadas en k-vecinos.
        
        Args:
            task_id: ID del nuevo dataset
            k_neighbors: Número de vecinos a considerar
            configs_per_neighbor: Configuraciones a extraer por vecino
            deduplicate: Si eliminar configuraciones duplicadas
            
        Returns:
            Lista de WarmStartConfig
        """
        # 1. Encontrar vecinos
        neighbors = self.get_neighbors(task_id, k=k_neighbors)
        
        logger.info(f"Vecinos encontrados para task {task_id}: {[n[0] for n in neighbors]}")
        
        all_configs = []
        
        # 2. Para cada vecino, obtener mejores configs
        for neighbor_id, distance in neighbors:
            # Obtener nombre del dataset
            dataset_name = self.reverse_mapping.get(neighbor_id)
            
            if dataset_name is None:
                # Buscar en historical_data directamente
                for ds in self.historical_data['Dataset'].unique():
                    if str(neighbor_id) in str(ds):
                        dataset_name = ds
                        break
            
            if dataset_name is None:
                continue
            
            # Obtener mejores configs
            configs = self.get_best_configs_for_dataset(dataset_name, n=configs_per_neighbor)
            
            for config in configs:
                warm_config = WarmStartConfig(
                    pipeline=config['pipeline'],
                    classifier=config['classifier'],
                    source_dataset=config['_source_dataset'],
                    source_accuracy=config['_source_accuracy'],
                    distance_to_new=distance
                )
                all_configs.append(warm_config)
        
        # 3. Deduplicar si es necesario
        if deduplicate:
            all_configs = self._deduplicate_configs(all_configs)
        
        # 4. Ordenar por distancia (más cercanos primero)
        all_configs.sort(key=lambda x: x.distance_to_new)
        
        return all_configs
    
    def _deduplicate_configs(
        self, 
        configs: List[WarmStartConfig]
    ) -> List[WarmStartConfig]:
        """Elimina configuraciones duplicadas."""
        seen = set()
        unique = []
        
        for config in configs:
            # Crear key basada en pipeline y classifier
            key = (
                tuple(sorted(config.pipeline.items())),
                tuple(sorted((k, str(v)) for k, v in config.classifier.items()))
            )
            
            if key not in seen:
                seen.add(key)
                unique.append(config)
        
        return unique
    
    def to_fsbo_format(
        self, 
        warm_configs: List[WarmStartConfig]
    ) -> List[Dict[str, Any]]:
        """
        Convierte WarmStartConfigs al formato esperado por FSBOPipelineOptimizer.
        
        Args:
            warm_configs: Lista de WarmStartConfig
            
        Returns:
            Lista de diccionarios {'pipeline': {...}, 'classifier': {...}}
        """
        return [
            {
                'pipeline': config.pipeline,
                'classifier': config.classifier
            }
            for config in warm_configs
        ]
    
    def suggest_initial(
        self,
        task_id: int,
        n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Sugiere n configuraciones iniciales para warm-start.
        
        INTERFAZ COMPATIBLE con FSBOPipelineOptimizer.suggest_initial()
        para facilitar comparación experimental.
        
        Estrategia: Obtiene las mejores configuraciones de los k datasets
        más similares (basado en embedding MetaFeatX).
        
        Args:
            task_id: ID del dataset objetivo
            n: Número de configuraciones a sugerir
            
        Returns:
            Lista de diccionarios {'pipeline': {...}, 'classifier': {...}}
        """
        # Calcular cuántos vecinos y configs por vecino necesitamos
        # para obtener aproximadamente n configs
        k_neighbors = min(n, 10)
        configs_per_neighbor = max(1, (n + k_neighbors - 1) // k_neighbors)
        
        warm_configs = self.get_warm_start_configs(
            task_id=task_id,
            k_neighbors=k_neighbors,
            configs_per_neighbor=configs_per_neighbor,
            deduplicate=True
        )
        
        # Limitar a n configs
        warm_configs = warm_configs[:n]
        
        return self.to_fsbo_format(warm_configs)
    
    def get_config_with_metadata(
        self,
        task_id: int,
        n: int = 5
    ) -> List[WarmStartConfig]:
        """
        Similar a suggest_initial pero retorna metadata adicional.
        
        Útil para análisis experimental (saber de qué dataset vino cada config).
        
        Args:
            task_id: ID del dataset objetivo
            n: Número de configuraciones
            
        Returns:
            Lista de WarmStartConfig con metadata (source_dataset, accuracy, distance)
        """
        k_neighbors = min(n, 10)
        configs_per_neighbor = max(1, (n + k_neighbors - 1) // k_neighbors)
        
        return self.get_warm_start_configs(
            task_id=task_id,
            k_neighbors=k_neighbors,
            configs_per_neighbor=configs_per_neighbor,
            deduplicate=True
        )[:n]


# =============================================================================
# CLI
# =============================================================================

def main():
    """Función principal para testing."""
    import argparse
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(
        description='Test MetaFeatXWarmStart - Warm-start basado en k-vecinos'
    )
    parser.add_argument('--algorithm', type=str, default='adaboost')
    parser.add_argument('--task-id', type=int, default=3)
    parser.add_argument('--n', type=int, default=5, help='Número de configs a sugerir')
    parser.add_argument('--data-path', type=str, default=None)
    parser.add_argument('--pipes-path', type=str, default=None)
    
    args = parser.parse_args()
    
    # Rutas
    base_path = Path(__file__).parent.parent.parent
    data_path = args.data_path or str(base_path / 'data')
    pipes_path = args.pipes_path or str(base_path / 'pipes' / 'combined')
    
    print("=" * 70)
    print("🔍 ESTRATEGIA: MetaFeatX Warm-Start (K-Vecinos)")
    print("=" * 70)
    print(f"\nConfiguración:")
    print(f"  Algoritmo: {args.algorithm}")
    print(f"  Task ID: {args.task_id}")
    print(f"  N configs: {args.n}")
    print(f"  Data path: {data_path}")
    print(f"  Pipes path: {pipes_path}")
    
    try:
        # Crear warm starter
        print("\n[1] Cargando datos...")
        warm_starter = MetaFeatXWarmStart.from_data(
            algorithm=args.algorithm,
            data_path=data_path,
            pipes_path=pipes_path
        )
        
        print(f"    ✓ Basic representations: {warm_starter.basic_representations.shape}")
        print(f"    ✓ Historical data: {warm_starter.historical_data.shape}")
        print(f"    ✓ Embedding: {warm_starter.embedding_matrix.shape}")
        
        # Método suggest_initial (interfaz compatible con FSBO)
        print(f"\n[2] suggest_initial(task_id={args.task_id}, n={args.n})...")
        configs = warm_starter.suggest_initial(task_id=args.task_id, n=args.n)
        
        print(f"    {len(configs)} configuraciones sugeridas:")
        for i, config in enumerate(configs, 1):
            print(f"\n    Config {i}:")
            print(f"      Pipeline: {config['pipeline']}")
            print(f"      Classifier: {config['classifier']}")
        
        # Método con metadata (para análisis)
        print(f"\n[3] get_config_with_metadata() - incluye info de origen...")
        configs_with_meta = warm_starter.get_config_with_metadata(
            task_id=args.task_id, 
            n=args.n
        )
        
        print(f"    Metadata de las configs:")
        for i, config in enumerate(configs_with_meta, 1):
            print(f"      {i}. Dataset: {config.source_dataset}, "
                  f"Accuracy: {config.source_accuracy:.4f}, "
                  f"Distancia: {config.distance_to_new:.4f}")
        
        print("\n" + "=" * 70)
        print("✅ Test completado")
        print("=" * 70)
        print("\nNota: Esta estrategia usa configs REALES de datasets similares.")
        print("      Comparar con FSBOPipelineOptimizer.suggest_initial()")
        print("      que usa predicciones del GP meta-entrenado.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
