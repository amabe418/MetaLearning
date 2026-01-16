# 🚀 Pipeline de Producción - Meta-Learning + FSBO

Sistema completo de optimización de hiperparámetros con transfer learning para uso en producción.

## 📋 Componentes

### 1. **Meta-Learner** (`meta_learner.py`)
Sugiere algoritmos basándose en:
- **Meta-features** del dataset
- **Knowledge Base** con historial de optimizaciones
- **Similitud** entre tareas

### 2. **Production Pipeline** (`production_pipeline.py`)
Pipeline completo que:
- Extrae meta-features automáticamente
- Usa meta-learner para sugerir algoritmos
- Optimiza con FSBO
- Guarda resultados en knowledge base

### 3. **Ejemplos** (`example_real_usage.py`)
5 ejemplos de uso con diferentes escenarios

---

## 🎯 Uso Rápido

```python
from production_pipeline import optimize_dataset
from sklearn.datasets import load_breast_cancer

# Cargar dataset
X, y = load_breast_cancer(return_X_y=True)

# Optimizar (una línea!)
result = optimize_dataset(
    X, y,
    dataset_id='breast_cancer',
    total_budget=100,
    max_algorithms=3
)

print(f"Mejor: {result.best_algorithm} con score {result.best_score:.3f}")
```

---

## 📚 Uso Avanzado

### Configuración Personalizada

```python
from production_pipeline import ProductionPipeline, ProductionConfig

# Configurar pipeline
config = ProductionConfig(
    total_budget=150,
    max_algorithms=5,
    similarity_threshold=0.4,      # Menos restrictivo
    min_similar_tasks=1,
    n_warm_start=10,
    early_stopping_patience=15,
    verbose=True
)

# Crear pipeline
pipeline = ProductionPipeline(config)

# Ejecutar
result = pipeline.run(
    X, y,
    dataset_id='my_dataset',
    test_size=0.2
)
```

### Evaluación Personalizada

```python
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

def custom_evaluation(algorithm, config, X_train, y_train, X_val, y_val):
    # Tu preprocesamiento
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    # Tu evaluación
    model = create_model(algorithm, config)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    # Tu métrica
    return f1_score(y_val, y_pred, average='weighted')

# Usar evaluación custom
result = pipeline.run(
    X, y,
    dataset_id='my_data',
    evaluation_fn=custom_evaluation
)
```

---

## 🧠 Cómo Funciona el Transfer Learning

### 1. **Knowledge Base**
El sistema mantiene un historial de optimizaciones:

```json
{
  "dataset_id": "iris",
  "meta_vector": [4.3, 2.0, 1.09, ...],
  "algorithm": "random_forest",
  "best_config": {"n_estimators": 100, ...},
  "best_score": 0.96
}
```

### 2. **Sugerencia de Algoritmos**

Cuando llega un nuevo dataset:

1. **Extrae meta-features** (tamaño, balance, complejidad, etc.)
2. **Busca tareas similares** en la KB usando distancia euclidiana + coseno
3. **Rankea algoritmos** por performance en tareas similares
4. **Devuelve top-k** con confianza

```python
# Meta-features del nuevo dataset
meta_vector = [4.5, 2.1, 1.15, 2.8, ...]

# Buscar similares en KB
similar_tasks = find_similar(meta_vector)
# → [("iris", similarity=0.92), ("wine", similarity=0.85), ...]

# Rankear algoritmos
# random_forest: 0.96, 0.94, 0.95 en tareas similares → score=0.95
# adaboost: 0.88, 0.90, 0.87 → score=0.88

# Sugerencia
→ ['random_forest', 'adaboost', 'gradient_boosting']
```

### 3. **Warm Start Inteligente**

Para cada algoritmo sugerido, usa configuraciones de tareas similares:

```python
# En lugar de empezar random:
configs = [random(), random(), random(), ...]

# Usa configs de tareas similares:
configs = [
    config_from_iris,      # similarity=0.92
    config_from_wine,      # similarity=0.85
    config_from_digits,    # similarity=0.78
    ...
]
```

---

## 📊 Análisis de Knowledge Base

```python
from pipeline import KnowledgeBase
from meta_learner import analyze_knowledge_base

kb = KnowledgeBase('experiments/knowledge_base.json')
stats = analyze_knowledge_base(kb)

print(f"Total de entradas: {stats['total_entries']}")
print(f"Datasets: {stats['datasets']}")

for algo, data in stats['algorithms'].items():
    print(f"{algo}: {data['count']} optimizaciones, "
          f"score promedio: {data['mean_score']:.3f}")
```

---

## 🔧 Configuración

### ProductionConfig

| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `total_budget` | Total de evaluaciones | 100 |
| `max_algorithms` | Máximo de algoritmos a optimizar | 5 |
| `similarity_threshold` | Umbral mínimo de similitud | 0.5 |
| `min_similar_tasks` | Mínimo de tareas similares | 2 |
| `n_warm_start` | Configs de warm start | 5 |
| `early_stopping_patience` | Paciencia para early stopping | 10 |

---

## 📈 Métricas de Confianza

La confianza se calcula como:

```python
confianza = (
    0.4 * (num_tareas_similares / 10) +  # Más tareas → más confianza
    0.4 * similitud_promedio +            # Mayor similitud → más confianza
    0.2 * consistencia                    # Menor varianza → más confianza
)
```

---

## 🎓 Ejemplos Completos

Ver `example_real_usage.py` para 5 ejemplos:

1. **Uso rápido** - Una línea de código
2. **Config personalizada** - Ajustar parámetros
3. **Evaluación custom** - Métrica propia
4. **Análisis KB** - Inspeccionar knowledge base
5. **OpenML** - Datasets externos

Ejecutar:
```bash
# Todos los ejemplos
python3 example_real_usage.py

# Ejemplo específico
python3 example_real_usage.py 1  # Ejemplo 1
```

---

## 🔄 Flujo Completo

```
1. Dataset Real (X, y)
   ↓
2. Extraer meta-features
   → [log(n_samples), n_features, class_balance, ...]
   ↓
3. Meta-Learner busca en KB
   → Tareas similares: iris (0.92), wine (0.85), ...
   ↓
4. Sugerir algoritmos
   → ['random_forest', 'adaboost', 'svm']
   ↓
5. Asignar presupuesto
   → random_forest: 40 evals (conf=0.85)
   → adaboost: 35 evals (conf=0.75)
   → svm: 25 evals (conf=0.60)
   ↓
6. Optimizar cada uno con FSBO
   → Warm start con configs de tareas similares
   → BO loop con early stopping
   ↓
7. Seleccionar mejor
   → random_forest: 0.94
   → adaboost: 0.91
   → svm: 0.88
   ↓
8. Guardar en KB
   → Nueva entrada para futuras optimizaciones
   ↓
9. Devolver resultado
   → {algorithm: 'random_forest', score: 0.94, config: {...}}
```

---

## 📦 Requisitos

```bash
pip install scikit-learn numpy torch gpytorch
```

---

## 🚨 Importante

### Primera Ejecución
Si la knowledge base está vacía, usa sugerencias por defecto:
- `random_forest` (confianza=0.7)
- `gradient_boosting` (confianza=0.65)
- `adaboost` (confianza=0.6)

### Construir Knowledge Base
Ejecuta optimizaciones en varios datasets para construir la KB:

```python
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, load_digits
)

datasets = [
    ('iris', load_iris()),
    ('wine', load_wine()),
    ('breast_cancer', load_breast_cancer()),
    ('digits', load_digits())
]

for name, data in datasets:
    optimize_dataset(data.data, data.target, name, total_budget=50)
```

---

## 🎯 Ventajas vs Pipeline de Prueba

| Aspecto | Pipeline Prueba | Pipeline Producción |
|---------|-----------------|---------------------|
| **Datos** | Sintéticos | Reales |
| **Meta-learner** | Simulado | Transfer Learning real |
| **Knowledge Base** | No se usa | Se usa y actualiza |
| **Evaluación** | Dummy | Modelos reales |
| **Warm start** | Aleatorio | De tareas similares |
| **Resultados** | No se guardan | Se persisten |

---

## 📝 Notas

- El sistema mejora con el uso (más datos en KB)
- La similitud de tareas es clave para el transfer learning
- Los meta-features capturan características del dataset
- El warm start acelera la convergencia significativamente

---

## 🤝 Contribuir

Para agregar nuevos algoritmos:

1. Entrenar modelo FSBO (ver `train_fsbo.py`)
2. Agregar configspace JSON
3. Agregar a `HYPERPARAMETER_SPACES` en `fsbo_optimizer.py`
4. Actualizar función de evaluación en pipeline

---

## 📧 Contacto

Proyecto: Meta-Learning + FSBO  
Autor: [Tu nombre]  
Fecha: Enero 2026

