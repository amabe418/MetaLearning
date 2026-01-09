"""
Synthetic Hyperparameter Configuration Generator
Basado en técnicas de: 
- BOPrO (Bayesian Optimization with Priors): Gaussian noise
- SMAC3 (Sequential Model-Based Algorithm Configuration): Surrogate models
- van Rijn & Hutter (2017): Hyperparameter sampling

Genera configuraciones sintéticas a partir de ejecuciones reales.
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Hiperparámetros numéricos a perturbar
NUMERIC_PARAMS = [
    'learning_rate',
    'batch_size',
    'weight_decay',
    'momentum',
    'dropout_rate',
    'alpha',
    'label_smoothing',
    'grad_clip'
]

# Límites para cada hiperparámetro (min, max)
PARAM_BOUNDS = {
    'learning_rate': (0.0001, 0.1),
    'batch_size': (16, 128),
    'weight_decay': (0.0, 0.01),
    'momentum': (0.0, 0.99),
    'dropout_rate': (0.0, 0.5),
    'alpha': (0.5, 1.0),
    'label_smoothing': (0.0, 0.3),
    'grad_clip': (0.0, 5.0),
}

# Tipo de distribución para cada parámetro
PARAM_TYPES = {
    'learning_rate': 'log',       # Escala logarítmica
    'batch_size': 'discrete',     # Entero (múltiplo de 8)
    'weight_decay':  'log',        # Escala logarítmica
    'momentum': 'uniform',        # Escala uniforme
    'dropout_rate': 'uniform',
    'alpha': 'uniform',
    'label_smoothing':  'uniform',
    'grad_clip': 'uniform',
}

# Métricas a interpolar
TARGET_METRICS = ['test_accuracy', 'train_accuracy', 'test_loss']

# ============================================================================
# FUNCIÓN 1: GENERACIÓN DE RUIDO GAUSSIANO (Basada en BOPrO)
# ============================================================================

def generate_gaussian_noise_configs(df_real, num_synthetic_per_config=5, noise_std=0.15, seed=42):
    """
    Genera configuraciones sintéticas añadiendo ruido gaussiano a las reales.
    
    Basado en: 
    - BOPrO (Balandat et al., 2020): https://arxiv.org/abs/2002.10389
    - SMAC3 (Hutter et al., 2011): Sampling strategies
    
    Args:
        df_real: DataFrame con configuraciones reales
        num_synthetic_per_config: Número de variaciones por cada config real
        noise_std: Desviación estándar del ruido (como % del valor)
        seed: Semilla para reproducibilidad
    
    Returns:
        DataFrame con configuraciones sintéticas (sin métricas interpoladas)
    """
    np.random.seed(seed)
    
    print(f"\n{'='*70}")
    print(f"🎲 GENERANDO CONFIGURACIONES SINTÉTICAS CON RUIDO GAUSSIANO")
    print(f"{'='*70}")
    print(f"Configuraciones reales: {len(df_real)}")
    print(f"Variaciones por config:  {num_synthetic_per_config}")
    print(f"Nivel de ruido (std): {noise_std}")
    
    synthetic_rows = []
    
    for idx, real_row in df_real.iterrows():
        for i in range(num_synthetic_per_config):
            synthetic_row = real_row.copy()
            
            # Perturbar cada hiperparámetro numérico
            for param in NUMERIC_PARAMS:
                if param not in real_row or pd.isna(real_row[param]):
                    continue
                
                original_value = float(real_row[param])
                param_type = PARAM_TYPES[param]
                min_val, max_val = PARAM_BOUNDS[param]
                
                # Generar ruido según el tipo de parámetro
                if param_type == 'log':
                    # Ruido en escala logarítmica (para LR, weight_decay)
                    if original_value > 0:
                        log_val = np.log10(original_value + 1e-10)
                        noise = np.random.normal(0, noise_std)
                        new_log_val = log_val + noise
                        new_value = 10 ** new_log_val
                    else:
                        new_value = original_value
                    
                elif param_type == 'discrete':
                    # Ruido para valores discretos (batch_size)
                    noise = np.random.normal(0, 16)  # std de 16
                    new_value = int(original_value + noise)
                    # Redondear a múltiplo de 8 (eficiencia GPU)
                    new_value = max(8, (new_value // 8) * 8)
                    
                else:  # 'uniform'
                    # Ruido proporcional al valor
                    if original_value > 0:
                        noise = np.random. normal(0, noise_std * original_value)
                        new_value = original_value + noise
                    else:
                        # Si es 0, aplicar ruido pequeño absoluto
                        noise = np.random.normal(0, noise_std * 0.1)
                        new_value = max(0, noise)
                
                # Aplicar límites
                new_value = np.clip(new_value, min_val, max_val)
                
                # Actualizar
                synthetic_row[param] = new_value
            
            # NO modificar las métricas todavía (se interpolarán después)
            for metric in TARGET_METRICS: 
                if metric in synthetic_row:
                    synthetic_row[metric] = np.nan
            
            synthetic_rows. append(synthetic_row)
    
    df_synthetic = pd.DataFrame(synthetic_rows)
    
    print(f"✓ Generadas {len(df_synthetic)} configuraciones sintéticas")
    print(f"{'='*70}\n")
    
    return df_synthetic

# ============================================================================
# FUNCIÓN 2: INTERPOLACIÓN CON SURROGATE MODEL (Basada en SMAC3)
# ============================================================================

def interpolate_metrics_with_surrogate(df_synthetic, df_real, method='knn', k=5):
    """
    Interpola métricas de configs sintéticas usando surrogate models.
    
    Basado en:
    - SMAC3: Random Forest surrogate
    - Optuna: K-NN + Tree-structured Parzen Estimator
    - van Rijn & Hutter (2017): Hyperparameter importance
    
    Args:
        df_synthetic: DataFrame con configs sintéticas (sin métricas)
        df_real: DataFrame con configs reales (con métricas)
        method: 'knn', 'random_forest', 'ensemble'
        k: Número de vecinos (para KNN)
    
    Returns:
        DataFrame sintético con métricas interpoladas
    """
    print(f"\n{'='*70}")
    print(f"🔄 INTERPOLANDO MÉTRICAS CON SURROGATE MODEL")
    print(f"{'='*70}")
    print(f"Método: {method. upper()}")
    print(f"Configuraciones reales: {len(df_real)}")
    print(f"Configuraciones sintéticas: {len(df_synthetic)}")
    
    # Preparar datos
    X_real = df_real[NUMERIC_PARAMS].fillna(0).values
    X_synthetic = df_synthetic[NUMERIC_PARAMS].fillna(0).values
    
    # Normalizar features
    scaler = StandardScaler()
    X_real_scaled = scaler.fit_transform(X_real)
    X_synthetic_scaled = scaler.transform(X_synthetic)
    
    # Interpolar cada métrica
    for metric in TARGET_METRICS: 
        if metric not in df_real.columns:
            print(f"  ⚠️  Métrica '{metric}' no encontrada, saltando...")
            continue
        
        y_real = df_real[metric].values
        
        # Elegir surrogate model
        if method == 'knn':
            # K-NN con ponderación por distancia
            model = KNeighborsRegressor(n_neighbors=k, weights='distance')
        
        elif method == 'random_forest':
            # Random Forest (como SMAC3)
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        
        elif method == 'ensemble':
            # Ensemble de KNN + RF (más robusto)
            from sklearn.ensemble import VotingRegressor
            knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
            rf = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
            model = VotingRegressor([('knn', knn), ('rf', rf)])
        
        else:
            raise ValueError(f"Método '{method}' no soportado")
        
        # Entrenar modelo
        model.fit(X_real_scaled, y_real)
        
        # Predecir para configs sintéticas
        y_synthetic = model.predict(X_synthetic_scaled)
        
        # Calcular confianza (validación cruzada en datos reales)
        cv_scores = cross_val_score(model, X_real_scaled, y_real, cv=3, scoring='r2')
        confidence = cv_scores.mean()
        
        # Actualizar DataFrame
        df_synthetic[metric] = y_synthetic
        
        print(f"  ✓ {metric: 20s} | R² CV: {confidence:.4f} | "
              f"Range: [{y_synthetic.min():.4f}, {y_synthetic.max():.4f}]")
    
    print(f"{'='*70}\n")
    
    return df_synthetic

# ============================================================================
# FUNCIÓN 3: VALIDACIÓN DE CONFIGS SINTÉTICAS
# ============================================================================

def validate_synthetic_configs(df_synthetic, df_real):
    """
    Valida que las configs sintéticas sean realistas comparando distribuciones.
    """
    print(f"\n{'='*70}")
    print(f"📊 VALIDACIÓN DE CONFIGURACIONES SINTÉTICAS")
    print(f"{'='*70}")
    
    for param in NUMERIC_PARAMS: 
        real_mean = df_real[param].mean()
        real_std = df_real[param].std()
        
        synth_mean = df_synthetic[param].mean()
        synth_std = df_synthetic[param].std()
        
        # Verificar que la distribución sintética sea similar a la real
        mean_diff = abs(synth_mean - real_mean) / real_mean if real_mean != 0 else 0
        std_diff = abs(synth_std - real_std) / real_std if real_std != 0 else 0
        
        status = "✓" if mean_diff < 0.3 and std_diff < 0.5 else "⚠️"
        
        print(f"  {status} {param: 20s} | "
              f"Mean:  {real_mean:.4f}→{synth_mean:.4f} ({mean_diff*100:+.1f}%) | "
              f"Std: {real_std:.4f}→{synth_std:. 4f} ({std_diff*100:+.1f}%)")
    
    print(f"{'='*70}\n")

# ============================================================================
# FUNCIÓN 4: ANÁLISIS COMPARATIVO
# ============================================================================

def comparative_analysis(df_real, df_synthetic, df_combined):
    """
    Genera estadísticas comparativas entre configs reales y sintéticas.
    """
    print(f"\n{'='*70}")
    print(f"📈 ANÁLISIS COMPARATIVO")
    print(f"{'='*70}")
    
    print(f"\n📌 RESUMEN:")
    print(f"  Configuraciones reales:      {len(df_real):,}")
    print(f"  Configuraciones sintéticas:  {len(df_synthetic):,}")
    print(f"  Total final:                {len(df_combined):,}")
    print(f"  Factor de aumento:          {len(df_combined)/len(df_real):.1f}x")
    
    print(f"\n📌 DISTRIBUCIÓN POR ARQUITECTURA:")
    for arch in df_combined['architecture'].unique():
        real_count = len(df_real[df_real['architecture'] == arch])
        synth_count = len(df_synthetic[df_synthetic['architecture'] == arch])
        print(f"  {arch:20s} | Real: {real_count:4d} | Sintético: {synth_count:4d}")
    
    print(f"\n📌 DISTRIBUCIÓN POR DATASET (primeros 10):")
    for task in df_combined['task_id'].unique()[:10]:
        real_count = len(df_real[df_real['task_id'] == task])
        synth_count = len(df_synthetic[df_synthetic['task_id'] == task])
        print(f"  {task:20s} | Real: {real_count:4d} | Sintético: {synth_count:4d}")
    
    print(f"\n📌 ESTADÍSTICAS DE MÉTRICAS:")
    for metric in TARGET_METRICS:
        if metric in df_combined.columns:
            print(f"\n  {metric}:")
            print(f"    Real      - Mean: {df_real[metric]. mean():.4f} | "
                  f"Std:  {df_real[metric].std():.4f} | "
                  f"Range: [{df_real[metric].min():.4f}, {df_real[metric].max():.4f}]")
            print(f"    Sintético - Mean:  {df_synthetic[metric].mean():.4f} | "
                  f"Std: {df_synthetic[metric].std():.4f} | "
                  f"Range: [{df_synthetic[metric].min():.4f}, {df_synthetic[metric].max():.4f}]")
    
    print(f"\n{'='*70}\n")

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def augment_hyperparameter_data(
    input_csv='target_representations.csv',
    output_csv='target_representations_augmented.csv',
    num_synthetic_per_config=5,
    noise_std=0.15,
    interpolation_method='knn',
    k_neighbors=5,
    seed=42
):
    """
    Pipeline completo para augmentar datos de hiperparámetros.
    
    Args:
        input_csv: CSV con configuraciones reales
        output_csv: CSV de salida con configs reales + sintéticas
        num_synthetic_per_config:  Cuántas variaciones generar por config real
        noise_std: Nivel de ruido gaussiano (0. 15 = 15%)
        interpolation_method: 'knn', 'random_forest', 'ensemble'
        k_neighbors:  Número de vecinos para KNN
        seed: Semilla para reproducibilidad
    
    Returns:
        df_combined: DataFrame con todos los datos
    """
    print(f"\n{'='*70}")
    print(f"🚀 PIPELINE DE AUGMENTACIÓN DE HIPERPARÁMETROS")
    print(f"{'='*70}")
    print(f"Input:   {input_csv}")
    print(f"Output: {output_csv}")
    print(f"Método: {interpolation_method. upper()}")
    print(f"{'='*70}\n")
    
    # 1. Cargar datos reales
    print("📂 Cargando configuraciones reales...")
    df_real = pd.read_csv(input_csv)
    print(f"✓ Cargadas {len(df_real)} configuraciones reales")
    print(f"✓ Datasets únicos: {df_real['task_id'].nunique()}")
    print(f"✓ Arquitecturas:  {df_real['architecture'].unique().tolist()}\n")
    
    # 2. Generar configs sintéticas con ruido gaussiano
    df_synthetic = generate_gaussian_noise_configs(
        df_real,
        num_synthetic_per_config=num_synthetic_per_config,
        noise_std=noise_std,
        seed=seed
    )
    
    # 3. Interpolar métricas con surrogate model
    df_synthetic = interpolate_metrics_with_surrogate(
        df_synthetic,
        df_real,
        method=interpolation_method,
        k=k_neighbors
    )
    
    # 4. Validar configs sintéticas
    validate_synthetic_configs(df_synthetic, df_real)
    
    # 5. Combinar real + sintético
    df_combined = pd. concat([df_real, df_synthetic], ignore_index=True)
    
    # 6. Añadir columna de identificación
    df_combined['is_synthetic'] = ['Real'] * len(df_real) + ['Synthetic'] * len(df_synthetic)
    
    # 7. Guardar
    df_combined.to_csv(output_csv, index=False)
    print(f"💾 Archivo guardado: {output_csv}")
    
    # 8. Análisis comparativo
    comparative_analysis(df_real, df_synthetic, df_combined)
    
    # 9. Resumen final
    print(f"\n{'='*70}")
    print(f"✅ PIPELINE COMPLETADO")
    print(f"{'='*70}")
    print(f"Total configuraciones:  {len(df_combined):,}")
    print(f"  - Reales:      {len(df_real):,} ({len(df_real)/len(df_combined)*100:.1f}%)")
    print(f"  - Sintéticas: {len(df_synthetic):,} ({len(df_synthetic)/len(df_combined)*100:.1f}%)")
    print(f"Factor de aumento: {len(df_combined)/len(df_real):.1f}x")
    print(f"{'='*70}\n")
    
    return df_combined

# ============================================================================
# SCRIPT PRINCIPAL
# ============================================================================

if __name__ == '__main__':
    # Configuración
    INPUT_CSV = 'target_representations.csv'
    OUTPUT_CSV = 'target_representations_augmented.csv'
    
    # Parámetros de augmentación
    NUM_SYNTHETIC_PER_CONFIG = 5   # 180 × 5 = 900 sintéticas → Total: 1,080
    NOISE_STD = 0.15               # 15% de desviación estándar
    INTERPOLATION_METHOD = 'knn'   # Opciones: 'knn', 'random_forest', 'ensemble'
    K_NEIGHBORS = 5                # Para KNN
    
    # Ejecutar pipeline
    df_augmented = augment_hyperparameter_data(
        input_csv=INPUT_CSV,
        output_csv=OUTPUT_CSV,
        num_synthetic_per_config=NUM_SYNTHETIC_PER_CONFIG,
        noise_std=NOISE_STD,
        interpolation_method=INTERPOLATION_METHOD,
        k_neighbors=K_NEIGHBORS,
        seed=42
    )
    
    # Mostrar preview
    print("\n📊 PREVIEW DEL DATASET AUGMENTADO:")
    print(df_augmented.head(15).to_string())
    
    print("\n📈 ESTADÍSTICAS POR TIPO:")
    print(df_augmented.groupby('is_synthetic')['test_accuracy'].describe())