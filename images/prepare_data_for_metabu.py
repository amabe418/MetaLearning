"""
Script para preparar los CSVs de imágenes para usar con Metabu.

Genera:
- metafeatures_ready.csv: meta-features normalizadas con task_id
- target_representations_ready.csv: configuraciones normalizadas y codificadas
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

print("="*80)
print("PREPARANDO DATOS PARA METABU")
print("="*80)

# ============================================================================
# 1. PROCESAR BASIC REPRESENTATIONS (metafeatures)
# ============================================================================
print("\n1. Procesando metafeatures_consistent.csv...")

basic = pd.read_csv("metafeatures_consistent.csv")
print(f"   - Cargadas {len(basic)} filas, {len(basic.columns)} columnas")

# Renombrar "dataset" → "task_id" para consistencia con target_representations
basic = basic.rename(columns={'dataset': 'task_id'})
print(f"   - Columna 'dataset' renombrada a 'task_id'")

# Detectar columnas categóricas en metafeatures
categorical_metafeatures = []
for col in basic.columns:
    if col == 'task_id':
        continue
    if basic[col].dtype == 'object' or basic[col].dtype.name == 'category':
        categorical_metafeatures.append(col)

if categorical_metafeatures:
    print(f"   - Columnas categóricas detectadas: {categorical_metafeatures}")
    # One-hot encoding
    basic = pd.get_dummies(basic, columns=categorical_metafeatures, 
                          prefix=categorical_metafeatures, dtype=float)
    print(f"   - One-hot encoding aplicado")

# Verificar NaN
nan_counts = basic.isnull().sum()
if nan_counts.sum() > 0:
    print(f"   - Valores NaN encontrados: {nan_counts[nan_counts > 0].to_dict()}")
    basic = basic.fillna(0)
    print(f"   - NaN rellenados con 0")
else:
    print(f"   - No hay valores NaN")

# Normalizar todas las columnas excepto task_id
cols_to_scale = [c for c in basic.columns if c != 'task_id']
print(f"   - Normalizando {len(cols_to_scale)} columnas (StandardScaler)...")

scaler_basic = StandardScaler()
basic[cols_to_scale] = scaler_basic.fit_transform(basic[cols_to_scale])

# Guardar
basic.to_csv("basic_representations.csv", index=False)
print(f"   ✓ Guardado en: basic_representations.csv")
print(f"   ✓ Shape: {basic.shape}")
print(f"   ✓ Datasets: {sorted(basic['task_id'].unique())}")


# ============================================================================
# 2. PROCESAR TARGET REPRESENTATIONS (configuraciones)
# ============================================================================
print("\n2. Procesando target_representations.csv...")

target = pd.read_csv("target_representations.csv")
print(f"   - Cargadas {len(target)} filas, {len(target.columns)} columnas")
print(f"   - Datasets: {sorted(target['task_id'].unique())}")

# Identificar arquitecturas únicas
architectures = sorted(target['architecture'].unique())
print(f"\n   ⚠️  ARQUITECTURAS DETECTADAS: {architectures}")
print(f"   → Metabu requiere UN CSV por algoritmo/arquitectura")
print(f"   → Generando {len(architectures)} CSVs separados...\n")

# Identificar tipos de columnas
id_col = 'task_id'
algo_col = 'architecture'
categorical_cols = ['optimizer']  # architecture ya no es categórica, es el separador
# Mantener test_accuracy como parte de la representación (SIN normalizar - ya está en [0,1])
cols_to_remove = ['train_accuracy', 'test_loss', 'training_time_sec']
accuracy_col = 'test_accuracy'  # Esta columna NO se normaliza

# Columnas numéricas de configuración (para normalizar)
config_cols = [c for c in target.columns 
               if c not in [id_col, algo_col, accuracy_col] + categorical_cols + cols_to_remove]

print(f"   - Columnas categóricas: {categorical_cols}")
print(f"   - Columnas de config a normalizar: {config_cols}")
print(f"   - Accuracy (NO se normaliza): {accuracy_col}")
print(f"   - Columnas a eliminar: {cols_to_remove}")

# PROCESAR CADA ARQUITECTURA POR SEPARADO
target_csvs = {}

for arch in architectures:
    print(f"\n   {'='*60}")
    print(f"   Procesando arquitectura: {arch}")
    print(f"   {'='*60}")
    
    # Filtrar solo esta arquitectura
    target_arch = target[target['architecture'] == arch].copy()
    print(f"   - Filas para {arch}: {len(target_arch)}")
    print(f"   - Datasets: {sorted(target_arch['task_id'].unique())}")
    
    # Eliminar columna 'architecture' y columnas no necesarias
    target_clean = target_arch.drop(columns=['architecture'] + cols_to_remove, errors='ignore')
    
    # Análisis de NaN en columnas de configuración
    nan_summary = []
    for col in config_cols:
        if col in target_clean.columns:
            nan_count = target_clean[col].isnull().sum()
            if nan_count > 0:
                nan_summary.append(f"{col}: {nan_count}")
                target_clean[col] = target_clean[col].fillna(0)
    
    if nan_summary:
        print(f"   - NaN rellenados: {', '.join(nan_summary)}")
    
    # One-hot encoding para categóricas (optimizer)
    if categorical_cols:
        target_encoded = pd.get_dummies(target_clean, 
                                       columns=categorical_cols, 
                                       prefix=categorical_cols,
                                       dtype=float)
        new_cols = [c for c in target_encoded.columns if c not in target_clean.columns]
        print(f"   - One-hot encoding: {new_cols}")
    else:
        target_encoded = target_clean
    
    # Normalizar solo columnas de configuración (NO test_accuracy)
    config_in_target = [c for c in config_cols if c in target_encoded.columns]
    if config_in_target:
        scaler_target = StandardScaler()
        target_encoded[config_in_target] = scaler_target.fit_transform(
            target_encoded[config_in_target]
        )
        print(f"   - Normalizadas {len(config_in_target)} columnas de config")
    
    # test_accuracy se mantiene sin normalizar
    if accuracy_col in target_encoded.columns:
        print(f"   - {accuracy_col} mantenida sin normalizar (rango [0, 1])")
    
    # Guardar
    filename = f"target_representations_{arch}.csv"
    target_encoded.to_csv(filename, index=False)
    target_csvs[arch] = filename
    print(f"   ✓ Guardado: {filename}")
    print(f"   ✓ Shape: {target_encoded.shape}")

print(f"\n   {'='*60}")
print(f"   ✓ Generados {len(target_csvs)} CSVs de target representations:")
for arch, filename in target_csvs.items():
    print(f"     - {filename}")
print(f"   {'='*60}")


# ============================================================================
# 3. VERIFICACIÓN FINAL
# ============================================================================
print("\n" + "="*80)
print("VERIFICACIÓN FINAL")
print("="*80)

# Verificar datasets comunes
basic_tasks = set(basic['task_id'])

print(f"\nDatasets en basic_representations.csv: {sorted(basic_tasks)}")

for arch, filename in target_csvs.items():
    target_arch = pd.read_csv(filename)
    target_tasks = set(target_arch['task_id'])
    common = basic_tasks & target_tasks
    
    print(f"\n{arch}:")
    print(f"  - Archivo: {filename}")
    print(f"  - Datasets: {sorted(target_tasks)}")
    print(f"  - Comunes con metafeatures: {len(common)} → {sorted(common)}")
    
    only_in_basic = basic_tasks - target_tasks
    only_in_target = target_tasks - basic_tasks
    if only_in_basic:
        print(f"  ⚠ Solo en metafeatures: {sorted(only_in_basic)}")
    if only_in_target:
        print(f"  ⚠ Solo en target: {sorted(only_in_target)}")

# Verificar NaN
print(f"\n{'='*80}")
print(f"NaN en basic_representations.csv: {basic.isnull().sum().sum()}")
for arch, filename in target_csvs.items():
    target_arch = pd.read_csv(filename)
    print(f"NaN en {filename}: {target_arch.isnull().sum().sum()}")

# Mostrar estructura final
print("\n" + "-"*80)
print("ESTRUCTURA FINAL:")
print("-"*80)

print("\nbasic_representations.csv:")
print(f"  Filas: {len(basic)}")
print(f"  Columnas: {len(basic.columns)}")
print(f"  Primeras columnas: {list(basic.columns[:5])} ... (total {len(basic.columns)})")

for arch, filename in target_csvs.items():
    target_arch = pd.read_csv(filename)
    print(f"\n{filename}:")
    print(f"  Filas: {len(target_arch)}")
    print(f"  Columnas: {len(target_arch.columns)}")
    print(f"  Columnas: {list(target_arch.columns)}")

print("\n" + "="*80)
print("✓ ARCHIVOS LISTOS PARA USAR CON METABU")
print("="*80)
print("\nUso (para cada arquitectura):")
print("""
from metabu import Metabu
import pandas as pd

# Ejemplo con ResNet18
basic = pd.read_csv("images/basic_representations.csv")
target = pd.read_csv("images/target_representations_ResNet18.csv")

# Filtrar solo datasets comunes
common = set(basic['task_id']) & set(target['task_id'])
basic_filtered = basic[basic['task_id'].isin(common)]
target_filtered = target[target['task_id'].isin(common)]

metabu = Metabu()
metabu.train(basic_reprs=basic_filtered, 
             target_reprs=target_filtered, 
             column_id='task_id')

# Repetir para EfficientNetB0 y MobileNetV2
""")
