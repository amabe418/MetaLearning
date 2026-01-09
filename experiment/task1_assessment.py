"""
Task 1 Assessment: Evaluación completa de topología definida por MetaFeatX

Basado en el paper METABU: "Learning Meta-Features for AutoML" (ICLR 2022)

Objetivo: Comparar los vecinos más cercanos basados en representaciones MetaFeatX 
con los vecinos basados en representaciones objetivo usando NDCG.

Metodología:
1. Leave-one-out evaluation: Para cada tarea, se entrena MetaFeatX con las demás tareas
2. Se calculan distancias entre tareas usando:
   - Representaciones MetaFeatX (euclidiana) -> ranking predicho
   - Representaciones objetivo (Wasserstein) -> ranking verdadero
3. Se compara usando NDCG@k
4. Se compara con baselines (hand-crafted meta-features)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import sys
import os

# Asegurar que podemos importar módulos del proyecto
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiment.utils import load_target_features
from model.utils import get_cost_matrix, get_ndcg_score
from model.metafeatx import MetaFeatX
from model.distance import distance_wasserstein


class MetaFeatureConfig:
    """Configuración simple para meta-features"""
    def __init__(self, name, basic_columns=None):
        self.name = name
        if basic_columns is None:
            # Por defecto, usar todas las columnas excepto task_id
            # Esto se ajustará cuando carguemos los datos
            self.basic_columns = "all"
        else:
            self.basic_columns = basic_columns


class PipelineConfig:
    """Configuración simple para pipeline/algoritmo"""
    def __init__(self, name):
        self.name = name


class TaskConfig:
    """Configuración simple para tarea"""
    def __init__(self, ndcg=10):
        self.ndcg = ndcg


class ExperimentConfig:
    """Configuración completa del experimento"""
    def __init__(self, data_path="data", seed=42, ndcg_values=[5, 10, 20]):
        self.data_path = data_path
        self.seed = seed
        self.task = TaskConfig()
        self.ndcg_values = ndcg_values


def evaluate_task1_single_task(
    test_task_id,
    train_task_ids,
    target_reprs,
    basic_reprs_all,
    metafeature_config,
    pipeline_config,
    cfg,
    use_metafeatx=True
):
    """
    Evalúa Task 1 para una sola tarea de test usando leave-one-out.
    
    Args:
        test_task_id: ID de la tarea de test
        train_task_ids: Lista de IDs de tareas de entrenamiento
        target_reprs: DataFrame con representaciones objetivo
        basic_reprs_all: DataFrame con todas las representaciones básicas
        metafeature_config: Configuración de meta-features
        pipeline_config: Configuración del pipeline/algoritmo
        cfg: Configuración del experimento
        use_metafeatx: Si True, usa MetaFeatX; si False, usa hand-crafted
    
    Returns:
        dict con resultados de NDCG para cada k
    """
    # Filtrar representaciones objetivo para train y test
    target_reprs_train = target_reprs[target_reprs['task_id'].isin(train_task_ids)].copy()
    target_reprs_all = target_reprs[target_reprs['task_id'].isin(train_task_ids + [test_task_id])].copy()
    
    # Obtener representaciones básicas o MetaFeatX
    if use_metafeatx:
        # Entrenar MetaFeatX y obtener representaciones aprendidas
        basic_reprs_subset = basic_reprs_all[basic_reprs_all['task_id'].isin(train_task_ids + [test_task_id])].copy()
        
        # Normalizar basic_reprs antes de entrenar MetaFeatX
        cols_to_scale = [c for c in basic_reprs_subset.columns if c != 'task_id']
        scaler = StandardScaler()
        basic_reprs_train_only = basic_reprs_subset[basic_reprs_subset['task_id'].isin(train_task_ids)]
        if len(basic_reprs_train_only) > 0:
            scaler.fit(basic_reprs_train_only[cols_to_scale])
            basic_reprs_normalized = basic_reprs_subset.copy()
            basic_reprs_normalized[cols_to_scale] = scaler.transform(basic_reprs_subset[cols_to_scale])
        else:
            basic_reprs_normalized = basic_reprs_subset.copy()
        
        # Entrenar MetaFeatX directamente (sin usar get_model_representations para evitar bootstrap)
        try:
            model = MetaFeatX(
                alpha=0.5,
                lambda_reg=1e-3,
                learning_rate=0.01,
                early_stopping_patience=20,
                early_stopping_criterion_ndcg=max(cfg.ndcg_values),
                verbose=False,
                seed=cfg.seed,
                ncpus=1
            )
            
            # Entrenar solo con train
            basic_reprs_train_meta = basic_reprs_normalized[basic_reprs_normalized['task_id'].isin(train_task_ids)].copy()
            target_reprs_train_meta = target_reprs[target_reprs['task_id'].isin(train_task_ids)].copy()
            
            model.train(
                basic_reprs=basic_reprs_train_meta,
                target_reprs=target_reprs_train_meta,
                column_id="task_id"
            )
            
            # Predecir para todas las tareas (train + test)
            basic_reprs_all_meta = basic_reprs_normalized[basic_reprs_normalized['task_id'].isin(train_task_ids + [test_task_id])].copy()
            learned_reprs = model.predict(basic_reprs_all_meta)
            
            # Crear DataFrame con representaciones aprendidas
            learned_reprs_df = pd.DataFrame(
                learned_reprs,
                columns=[f"dim_{i}" for i in range(learned_reprs.shape[1])]
            )
            learned_reprs_df['task_id'] = basic_reprs_all_meta['task_id'].values
            
            feature_reprs = learned_reprs_df.set_index('task_id')
            feature_type = "metafeatx"
            
        except Exception as e:
            print(f"    Error entrenando MetaFeatX para task {test_task_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    else:
        # Usar hand-crafted meta-features directamente
        basic_reprs = basic_reprs_all[basic_reprs_all['task_id'].isin(train_task_ids + [test_task_id])].copy()
        
        # Normalizar (usar solo train para fit, aplicar a todas)
        cols_to_scale = [c for c in basic_reprs.columns if c != 'task_id']
        scaler = StandardScaler()
        basic_reprs_train = basic_reprs[basic_reprs['task_id'].isin(train_task_ids)]
        if len(basic_reprs_train) > 0:
            scaler.fit(basic_reprs_train[cols_to_scale])
            basic_reprs_scaled = basic_reprs.copy()
            basic_reprs_scaled[cols_to_scale] = scaler.transform(basic_reprs[cols_to_scale])
        else:
            basic_reprs_scaled = basic_reprs.copy()
        
        feature_reprs = basic_reprs_scaled.set_index('task_id')
        feature_type = "handcrafted"
    
    # Calcular matriz de distancias objetivo (Wasserstein) - solo para todas las tareas
    all_task_ids = train_task_ids + [test_task_id]
    true_dist = get_cost_matrix(
        target_repr=target_reprs_all,
        task_ids=all_task_ids,
        verbose=False,
        column_id="task_id",
        pairwise_target_dist_func=distance_wasserstein,
        ncpus=1
    )
    
    # Calcular matriz de distancias de features (euclidiana)
    pred_dist = pairwise_distances(
        feature_reprs.loc[all_task_ids],
        metric="euclidean"
    )
    
    # Encontrar índice de la tarea de test
    test_idx = all_task_ids.index(test_task_id)
    
    # Calcular NDCG para cada k
    results = {
        'task_id': test_task_id,
        'feature_type': feature_type,
        'pipeline': pipeline_config.name
    }
    
    for k in cfg.ndcg_values:
        ndcg = get_ndcg_score(
            dist_pred=np.array([pred_dist[test_idx]]),
            dist_true=np.array([true_dist[test_idx]]),
            k=k
        )
        results[f'ndcg@{k}'] = ndcg
    
    return results


def run_task1_assessment(cfg, pipeline_names=['adaboost', 'random_forest', 'libsvm_svc'], 
                         metafeature_configs=None):
    """
    Ejecuta evaluación completa del Task 1 para múltiples algoritmos y meta-features.
    
    Args:
        cfg: ExperimentConfig con configuración del experimento
        pipeline_names: Lista de nombres de algoritmos a evaluar
        metafeature_configs: Lista de configuraciones de meta-features a comparar.
                            Si None, usa MetaFeatX y hand-crafted básico
    
    Returns:
        pd.DataFrame con todos los resultados
    """
    if metafeature_configs is None:
        # Configuración por defecto: comparar MetaFeatX vs hand-crafted
        metafeature_configs = [
            MetaFeatureConfig(name="model_metafeatx"),
            MetaFeatureConfig(name="handcrafted")
        ]
    
    all_results = []
    
    print("="*70)
    print("TASK 1 ASSESSMENT: Evaluación de Topología MetaFeatX")
    print("="*70)
    print(f"Algoritmos a evaluar: {pipeline_names}")
    print(f"Métricas NDCG@k: {cfg.ndcg_values}")
    print(f"Seed: {cfg.seed}")
    print()
    
    # Cargar representaciones básicas una vez
    print("Cargando representaciones básicas...")
    basic_reprs_all = pd.read_csv(f"{cfg.data_path}/basic_representations.csv").fillna(0)
    print(f"  Tareas disponibles: {len(basic_reprs_all['task_id'].unique())}")
    
    # Determinar columnas de basic_reprs para configurar metafeature
    basic_cols = [c for c in basic_reprs_all.columns if c != 'task_id']
    for mf_cfg in metafeature_configs:
        if mf_cfg.basic_columns == "all":
            mf_cfg.basic_columns = ",".join(basic_cols)
    
    # Iterar sobre cada algoritmo
    for pipeline_name in pipeline_names:
        print("\n" + "="*70)
        print(f"ALGORITMO: {pipeline_name.upper()}")
        print("="*70)
        
        pipeline_config = PipelineConfig(name=pipeline_name)
        
        try:
            # Cargar representaciones objetivo
            target_reprs = load_target_features(pipeline=pipeline_config, path=cfg.data_path)
            task_ids = sorted(list(target_reprs['task_id'].unique()))
            print(f"Tareas con representaciones objetivo: {len(task_ids)}")
            
            # Filtrar basic_reprs para tareas que existen en target
            common_task_ids = sorted(list(
                set(basic_reprs_all['task_id'].unique()) & set(task_ids)
            ))
            print(f"Tareas comunes (basic + target): {len(common_task_ids)}")
            
            if len(common_task_ids) < 2:
                print(f"  ⚠️  Insuficientes tareas comunes. Saltando {pipeline_name}")
                continue
            
            # Iterar sobre cada configuración de meta-features
            for mf_cfg in metafeature_configs:
                print(f"\n  Meta-features: {mf_cfg.name}")
                use_metafeatx = (mf_cfg.name == "model_metafeatx")
                
                # Leave-one-out evaluation
                task_results = []
                for test_task_id in tqdm(common_task_ids, desc=f"    Evaluando {mf_cfg.name}"):
                    train_task_ids = [tid for tid in common_task_ids if tid != test_task_id]
                    
                    try:
                        result = evaluate_task1_single_task(
                            test_task_id=test_task_id,
                            train_task_ids=train_task_ids,
                            target_reprs=target_reprs,
                            basic_reprs_all=basic_reprs_all,
                            metafeature_config=mf_cfg,
                            pipeline_config=pipeline_config,
                            cfg=cfg,
                            use_metafeatx=use_metafeatx
                        )
                        
                        if result is not None:
                            task_results.append(result)
                    except Exception as e:
                        print(f"\n    ⚠️  Error en task {test_task_id}: {e}")
                        continue
                
                if task_results:
                    all_results.extend(task_results)
                    # Mostrar resumen
                    df_temp = pd.DataFrame(task_results)
                    print(f"\n    Resumen {mf_cfg.name}:")
                    for k in cfg.ndcg_values:
                        mean_ndcg = df_temp[f'ndcg@{k}'].mean()
                        std_ndcg = df_temp[f'ndcg@{k}'].std()
                        print(f"      NDCG@{k}: {mean_ndcg:.4f} ± {std_ndcg:.4f}")
        
        except FileNotFoundError as e:
            print(f"  ⚠️  No se encontraron representaciones objetivo para {pipeline_name}: {e}")
            continue
        except Exception as e:
            print(f"  ⚠️  Error procesando {pipeline_name}: {e}")
            continue
    
    # Combinar todos los resultados
    if all_results:
        results_df = pd.DataFrame(all_results)
        print("\n" + "="*70)
        print("EVALUACIÓN COMPLETADA")
        print("="*70)
        return results_df
    else:
        print("\n⚠️  No se generaron resultados")
        return pd.DataFrame()


if __name__ == "__main__":
    import os
    import sys
    
    # Determinar path base del proyecto (un nivel arriba de experiment/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, "data")
    
    # Agregar project_root al path para imports
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    print(f"Project root: {project_root}")
    print(f"Data path: {data_path}")
    print()
    
    # Configuración del experimento
    cfg = ExperimentConfig(
        data_path=data_path,
        seed=42,
        ndcg_values=[5, 10, 20]
    )
    
    # Ejecutar evaluación
    results = run_task1_assessment(
        cfg=cfg,
        pipeline_names=['adaboost', 'random_forest', 'libsvm_svc']
    )
    
    if not results.empty:
        # Guardar resultados en el directorio del script
        output_file = os.path.join(script_dir, "task1_assessment_results.csv")
        results.to_csv(output_file, index=False)
        print(f"\n✓ Resultados guardados en: {output_file}")
        
        # Mostrar resumen por algoritmo y tipo de meta-feature
        print("\n" + "="*70)
        print("RESUMEN POR ALGORITMO Y TIPO DE META-FEATURE")
        print("="*70)
        for pipeline in sorted(results['pipeline'].unique()):
            print(f"\n{pipeline.upper()}:")
            for feature_type in sorted(results['feature_type'].unique()):
                subset = results[(results['pipeline'] == pipeline) & 
                                (results['feature_type'] == feature_type)]
                if not subset.empty:
                    print(f"  {feature_type}:")
                    for k in cfg.ndcg_values:
                        mean_ndcg = subset[f'ndcg@{k}'].mean()
                        std_ndcg = subset[f'ndcg@{k}'].std()
                        n = len(subset)
                        print(f"    NDCG@{k}: {mean_ndcg:.4f} ± {std_ndcg:.4f} (n={n})")
        
        # Comparación directa MetaFeatX vs Hand-crafted
        print("\n" + "="*70)
        print("COMPARACIÓN: MetaFeatX vs Hand-crafted")
        print("="*70)
        for pipeline in sorted(results['pipeline'].unique()):
            print(f"\n{pipeline.upper()}:")
            metafeatx_subset = results[(results['pipeline'] == pipeline) & 
                                       (results['feature_type'] == 'metafeatx')]
            handcrafted_subset = results[(results['pipeline'] == pipeline) & 
                                         (results['feature_type'] == 'handcrafted')]
            
            if not metafeatx_subset.empty and not handcrafted_subset.empty:
                for k in cfg.ndcg_values:
                    mf_mean = metafeatx_subset[f'ndcg@{k}'].mean()
                    hc_mean = handcrafted_subset[f'ndcg@{k}'].mean()
                    improvement = mf_mean - hc_mean
                    print(f"  NDCG@{k}:")
                    print(f"    MetaFeatX:    {mf_mean:.4f}")
                    print(f"    Hand-crafted: {hc_mean:.4f}")
                    print(f"    Mejora:       {improvement:+.4f} ({improvement/hc_mean*100:+.2f}%)")
    else:
        print("\n⚠️  No se generaron resultados. Revisa los errores anteriores.")
