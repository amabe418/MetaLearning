# Pipeline Completo para Metabu con Imágenes

## 🚀 Ejecución Rápida

```bash
cd images/
./run_full_pipeline.sh
```

Este script ejecuta automáticamente todo el pipeline:

1. ⬇️ **Descarga de datasets** (MNIST, CIFAR10, FashionMNIST, etc.)
2. 🔍 **Extracción de meta-features** (características visuales, estadísticas, etc.)
3. 🧠 **Entrenamiento de redes neuronales** (ResNet18, EfficientNetB0, MobileNetV2)
4. 📊 **Preparación de CSVs para Metabu** (normalización, one-hot encoding, etc.)

---

## 📋 Requisitos Previos

### Dependencias Python

```bash
pip install torch torchvision pandas scikit-learn tqdm numpy pillow
```

### Opcional: GPU para acelerar entrenamiento

Si tienes GPU NVIDIA con CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 🔧 Ejecución Paso a Paso (Manual)

Si prefieres ejecutar cada paso por separado:

### Paso 1: Descargar Datasets

```bash
python3 download_all_datasets.py
```

**Salida esperada:**
- `data/MNIST/`
- `data/CIFAR10/`
- `data/FashionMNIST/`
- ... (otros datasets)

---

### Paso 2: Extraer Meta-features

```bash
python3 extract_metafeatures_descargados.py
```

**Salida esperada:**
- `metafeatures_consistent.csv` (~13 datasets × ~40 features)

---

### Paso 3: Entrenar Redes Neuronales

```bash
python3 generate_target_representations.py
# O la versión alternativa:
# python3 generate_target_representations_VOL2.py
```

**⚠️ ADVERTENCIA:** Este paso puede tomar **horas** dependiendo de:
- CPU/GPU disponible
- Número de datasets
- Número de configuraciones a probar

**Salida esperada:**
- `target_representations.csv` (~96 configuraciones)

---

### Paso 4: Preparar CSVs para Metabu

```bash
python3 prepare_data_for_metabu.py
```

**Salida esperada:**
- `basic_representations.csv` (meta-features normalizadas)
- `target_representations_ResNet18.csv`
- `target_representations_EfficientNetB0.csv`
- `target_representations_MobileNetV2.csv`

---

## 📂 Archivos Generados

```
images/
├── data/                                    # Datasets descargados
│   ├── MNIST/
│   ├── CIFAR10/
│   └── ...
│
├── metafeatures_consistent.csv              # Meta-features originales
├── target_representations.csv               # Ejecuciones originales
│
├── basic_representations.csv                # ✅ LISTO PARA METABU
├── target_representations_ResNet18.csv      # ✅ LISTO PARA METABU
├── target_representations_EfficientNetB0.csv# ✅ LISTO PARA METABU
└── target_representations_MobileNetV2.csv   # ✅ LISTO PARA METABU
```

---

## 💻 Uso con Metabu

Una vez ejecutado el pipeline, usa los CSVs generados:

```python
from metabu import Metabu
import pandas as pd

# Cargar datos
basic = pd.read_csv("images/basic_representations.csv")
target = pd.read_csv("images/target_representations_ResNet18.csv")

# Filtrar solo datasets comunes
common = set(basic['task_id']) & set(target['task_id'])
basic_filtered = basic[basic['task_id'].isin(common)]
target_filtered = target[target['task_id'].isin(common)]

print(f"Entrenando con {len(common)} datasets: {sorted(common)}")

# Entrenar Metabu
metabu = Metabu(verbose=True)
metabu.train(
    basic_reprs=basic_filtered,
    target_reprs=target_filtered,
    column_id='task_id'
)

# Predecir embeddings
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

## 🔍 Verificación de Archivos

Para verificar que los archivos se generaron correctamente:

```bash
# Ver estructura de basic_representations.csv
head -2 basic_representations.csv

# Ver cuántos datasets hay
tail -n +2 basic_representations.csv | wc -l

# Ver estructura de target_representations
head -2 target_representations_ResNet18.csv

# Ver cuántas configuraciones hay
tail -n +2 target_representations_ResNet18.csv | wc -l
```

---

## ⏱️ Tiempos Estimados

| Paso | CPU | GPU (CUDA) |
|------|-----|------------|
| 1. Descargar datasets | ~10-30 min | ~10-30 min |
| 2. Extraer meta-features | ~5-15 min | ~5-15 min |
| 3. Entrenar redes neuronales | **~4-8 horas** | **~30-60 min** |
| 4. Preparar CSVs | ~5-10 seg | ~5-10 seg |
| **TOTAL** | **~5-9 horas** | **~1 hora** |

---

## 🛠️ Troubleshooting

### Error: "CUDA out of memory"

Si tienes GPU pero falla por falta de memoria:
- Reduce el batch_size en el script de entrenamiento
- O ejecuta en CPU (más lento pero funciona)

### Error: "Module not found"

```bash
pip install torch torchvision pandas scikit-learn tqdm
```

### Los archivos ya existen

El script detecta archivos existentes y los usa en lugar de regenerarlos. Si quieres regenerar todo:

```bash
# Eliminar archivos intermedios
rm -f metafeatures_consistent.csv target_representations.csv

# Eliminar archivos finales
rm -f basic_representations.csv target_representations_*.csv

# Re-ejecutar pipeline
./run_full_pipeline.sh
```

---

## 📊 Estructura de los CSVs Finales

### basic_representations.csv (49 columnas)

- `task_id`: Identificador del dataset
- `landmarker_*`: Accuracy con clasificadores simples
- `visual_variability`, `texture_contrast`, etc.: Features visuales
- `pixel_mean`, `pixel_std`, etc.: Estadísticas de píxeles
- `dataset_type_*`, `resolution_*`: Variables categóricas (one-hot)

### target_representations_[arquitectura].csv (12 columnas)

- `task_id`: Identificador del dataset
- `learning_rate`, `batch_size`, etc.: Hiperparámetros (normalizados)
- `test_accuracy`: Accuracy obtenida (**SIN normalizar**, rango [0, 1])
- `optimizer_Adam`: Optimizador usado (one-hot)

---

## 📝 Notas

1. **test_accuracy NO está normalizada** en los target_representations porque ya está en [0, 1]
2. Los hiperparámetros SÍ están normalizados con StandardScaler
3. Variables categóricas están en one-hot encoding
4. Valores NaN en hiperparámetros opcionales se rellenan con 0
5. Se generan 3 archivos target separados (uno por arquitectura) porque Metabu espera un CSV por algoritmo
