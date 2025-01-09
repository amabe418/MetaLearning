# Preparación de Datos de Imágenes para Metabu

## ⚠️ Importante: Un CSV por Arquitectura

**Metabu requiere UN modelo por algoritmo.** Por eso generamos **3 archivos separados**:
- `target_representations_ResNet18_ready.csv`
- `target_representations_EfficientNetB0_ready.csv`
- `target_representations_MobileNetV2_ready.csv`

Cada uno representa configuraciones de UNA arquitectura en diferentes datasets.

---

## Archivos Generados

### ✅ metafeatures_ready.csv
- **Filas:** 13 datasets de imágenes
- **Columnas:** 49 (1 task_id + 48 meta-features normalizadas)
- **Procesamiento aplicado:**
  - ✓ Columna `dataset` renombrada a `task_id`
  - ✓ Columnas categóricas (`dataset_type`, `resolution`) → one-hot encoding
  - ✓ Todas las features numéricas normalizadas con StandardScaler
  - ✓ Sin valores NaN

**Datasets incluidos:**
- CIFAR10, CIFAR100, DTD, EMNIST, EuroSAT
- FashionMNIST, Flowers102, GTSRB, MNIST
- Omniglot, OxfordIIITPet, SVHN, USPS

---

### ✅ target_representations_[ARQUITECTURA]_ready.csv

**3 archivos generados (uno por arquitectura):**

1. **target_representations_ResNet18_ready.csv**
2. **target_representations_EfficientNetB0_ready.csv**
3. **target_representations_MobileNetV2_ready.csv**

- **Filas:** 32 configuraciones por archivo
- **Columnas:** 11 (1 task_id + 10 features de configuración)
- **Procesamiento aplicado:**
  - ✓ Eliminadas columnas de resultados: `test_accuracy`, `train_accuracy`, `test_loss`, `training_time_sec`
  - ✓ Eliminada columna `architecture` (ya está implícita en el nombre del archivo)
  - ✓ Hiperparámetro categórico → one-hot:
    - `optimizer`: Adam (siempre 1.0 en estos CSVs)
  - ✓ Hiperparámetros numéricos normalizados:
    - `learning_rate`, `batch_size`, `weight_decay`, `momentum`
    - `dropout_rate`, `alpha`, `label_smoothing`, `grad_clip`, `epochs`
  - ✓ Valores NaN rellenados con 0 antes de normalizar
  - ✓ Sin valores NaN

**Datasets incluidos en cada archivo:**
- CIFAR10, CIFAR100, DTD, EMNIST_Balanced, EMNIST_ByClass
- EMNIST_ByMerge, EMNIST_Digits, EMNIST_MNIST, EuroSAT
- FashionMNIST, Flowers102, KMNIST, MNIST, QMNIST
- RenderedSST2, USPS

---

## ⚠️ Advertencia: Datasets Desbalanceados

**Problema detectado:**

```
Datasets SOLO en metafeatures (sin ejecuciones):
- EMNIST, SVHN, Omniglot, GTSRB, OxfordIIITPet

Datasets SOLO en targets (sin metafeatures):
- EMNIST_Balanced, EMNIST_ByClass, EMNIST_ByMerge
- EMNIST_Digits, EMNIST_MNIST
- KMNIST, QMNIST, RenderedSST2
```

**Datasets comunes en TODOS los archivos (8):**
- CIFAR10, CIFAR100, DTD, EuroSAT
- FashionMNIST, Flowers102, MNIST, USPS

### Por qué entrenar 3 modelos Metabu separados:

Como en el ejemplo de AdaBoost, Metabu aprende **similitud entre datasets PARA UN ALGORITMO específico**.

- **Metabu_ResNet18:** aprende qué datasets se comportan similar bajo ResNet18
- **Metabu_EfficientNetB0:** aprende similitud específica para EfficientNetB0
- **Metabu_MobileNetV2:** aprende similitud específica para MobileNetV2

Cada arquitectura puede tener una "noción" diferente de qué hace a dos datasets similares.

---

## Uso con Metabu

```python
import pandas as pd
from metabu import Metabu

# Cargar datos preparados
basic = pd.read_csv("images/metafeatures_ready.csv")
target = pd.read_csv("images/target_representations_ready.csv")

# Filtrar solo datasets comunes (recomendado)
common_tasks = set(basic['task_id']) & set(target['task_id'])
basic_filtered = basic[basic['task_id'].isin(common_tasks)]
target_filtered = target[target['task_id'].isin(common_tasks)]

print(f"Entrenando con {len(common_tasks)} datasets: {sorted(common_tasks)}")

# Entrenar Metabu
metabu = Metabu(verbose=True)
metabu.train(
    basic_reprs=basic_filtered,
    target_reprs=target_filtered,
    column_id='task_id'
)

# Obtener meta-features aprendidas
metabu_features = metabu.predict(basic_filtered)
print(f"Metabu features shape: {metabu_features.shape}")

# Ver importancias
importances, labels = metabu.get_importances()
top_features = sorted(zip(labels, importances), 
                     key=lambda x: x[1], reverse=True)[:10]
print("\nTop 10 meta-features más importantes:")
for feat, imp in top_features:
    print(f"  {feat}: {imp:.4f}")
```

---

## Regenerar los CSVs

Si necesitas regenerar los archivos procesados:

```bash
cd images/
python3 prepare_data_for_metabu.py
```

El script `prepare_data_for_metabu.py` lee los CSVs originales y genera automáticamente las versiones `_ready.csv`.

---

## Estructura de Columnas

### metafeatures_ready.csv (49 columnas)

```
task_id                          # ID del dataset
landmarker_1nn                   # Accuracy con 1-NN
landmarker_lda                   # Accuracy con LDA
landmarker_svm                   # Accuracy con SVM
landmarker_tree                  # Accuracy con DecisionTree
landmarker_nb                    # Accuracy con Naive Bayes
dataset_difficulty               # Dificultad estimada
visual_variability               # Variabilidad visual
structural_regularity            # Regularidad estructural
class_entropy                    # Entropía de clases
imbalance_ratio                  # Ratio de desbalance
class_overlap_index              # Índice de overlap
n_classes                        # Número de clases
color_entropy_mean               # Entropía de color promedio
color_diversity                  # Diversidad de color
color_saturation                 # Saturación
color_contrast                   # Contraste de color
texture_contrast                 # Contraste de textura
texture_homogeneity              # Homogeneidad de textura
texture_energy                   # Energía de textura
texture_correlation              # Correlación de textura
texture_complexity               # Complejidad de textura
edge_density                     # Densidad de bordes
edge_strength                    # Fuerza de bordes
shape_complexity                 # Complejidad de formas
spatial_compactness              # Compacidad espacial
aspect_ratio_variation           # Variación de aspect ratio
num_train                        # Número de muestras de entrenamiento
channels                         # Canales (1=grayscale, 3=RGB)
total_size_mb                    # Tamaño total en MB
pixel_mean                       # Media de píxeles
pixel_std                        # Desviación estándar de píxeles
pixel_entropy                    # Entropía de píxeles
pixel_skewness                   # Asimetría de píxeles
pixel_kurtosis                   # Curtosis de píxeles
pixel_gmean                      # Media geométrica
pixel_iqr                        # Rango intercuartil
pixel_median                     # Mediana de píxeles
dataset_type_color               # One-hot: es color
dataset_type_grayscale           # One-hot: es grayscale
resolution_28x28                 # One-hot: resolución 28x28
resolution_32x32                 # One-hot: resolución 32x32
resolution_48x48                 # One-hot: resolución 48x48
resolution_64x64                 # One-hot: resolución 64x64
resolution_96x96                 # One-hot: resolución 96x96
resolution_224x224               # One-hot: resolución 224x224
resolution_227x227               # One-hot: resolución 227x227
resolution_variablesize          # One-hot: tamaño variable
```

### target_representations_ready.csv (14 columnas)

```
task_id                          # ID del dataset
learning_rate                    # Learning rate (normalizado)
batch_size                       # Batch size (normalizado)
weight_decay                     # Weight decay (normalizado)
momentum                         # Momentum (normalizado, 0 si no aplica)
dropout_rate                     # Dropout rate (normalizado, 0 si no aplica)
alpha                            # Alpha (normalizado)
label_smoothing                  # Label smoothing (normalizado, 0 si no aplica)
grad_clip                        # Gradient clipping (normalizado, 0 si no aplica)
epochs                           # Número de epochs (normalizado)
architecture_EfficientNetB0      # One-hot: usa EfficientNetB0
architecture_MobileNetV2         # One-hot: usa MobileNetV2
architecture_ResNet18            # One-hot: usa ResNet18
optimizer_Adam                   # One-hot: usa Adam (siempre 1.0 en este caso)
```

---

## Notas Técnicas

1. **StandardScaler:** Todas las features numéricas están centradas (media=0) y escaladas (std=1)

2. **One-hot encoding:** Variables categóricas convertidas a binarias (0.0 o 1.0)

3. **NaN handling:** Valores faltantes rellenados con 0 ANTES de normalizar

4. **Eliminación de resultados:** Las columnas de accuracy/loss se eliminaron porque son OUTPUTS del entrenamiento, no parte de la configuración

5. **Consistencia:** Ambos CSVs usan `task_id` como columna identificadora
