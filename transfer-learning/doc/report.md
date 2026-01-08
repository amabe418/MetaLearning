# Reporte: FSBO Transfer Learning para Optimización de Hiperparámetros

## 1. Introducción

### 1.1 Contexto del Proyecto

Este proyecto implementa la parte de **Transfer Learning** de un sistema de AutoML compuesto por dos módulos:

```
┌─────────────────────────────────────────────────────────────────────┐
│                      SISTEMA AUTOML COMPLETO                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ENTRADA: Nuevo dataset                                            │
│                │                                                    │
│                ▼                                                    │
│   ┌───────────────────────────────────┐                            │
│   │        META-LEARNING              │                            │
│   │   "¿Qué algoritmos usar?"         │                            │
│   │                                   │                            │
│   │   Analiza meta-features →         │                            │
│   │   Sugiere: [RF, SVM, AdaBoost]    │                            │
│   └───────────────────────────────────┘                            │
│                │                                                    │
│                ▼                                                    │
│   ┌───────────────────────────────────┐                            │
│   │     TRANSFER-LEARNING (FSBO)      │  ← Este módulo             │
│   │   "¿Qué hiperparámetros usar?"    │                            │
│   │                                   │                            │
│   │   Optimiza HP para cada algoritmo │                            │
│   │   con pocas evaluaciones          │                            │
│   └───────────────────────────────────┘                            │
│                │                                                    │
│                ▼                                                    │
│   SALIDA: Mejor (algoritmo, configuración) para el dataset         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 FSBO (Few-Shot Bayesian Optimization)

FSBO es una técnica de meta-learning propuesta por Wistuba & Grabocka (ICLR 2021) que permite encontrar buenas configuraciones de hiperparámetros con muy pocas evaluaciones mediante:

1. **Deep Kernel GP**: Un Gaussian Process con kernel aprendido por una red neuronal
2. **Meta-Learning**: Pre-entrenar en muchas tareas fuente
3. **Transfer**: Adaptar rápidamente a nuevas tareas

---

## 2. Implementación

### 2.1 Arquitectura del Modelo

```
         x (hiperparámetros)
              │
              ▼
    ┌─────────────────┐
    │ DeepKernelNetwork│   φ(x): Red neuronal
    │   (2 capas, 128) │   Transforma HP a espacio latente
    └────────┬────────┘
              │
              ▼
         φ(x) ∈ ℝ¹²⁸
              │
              ▼
    ┌─────────────────┐
    │   RBF Kernel    │   k(φ(x), φ(x'))
    │   con ARD       │   Similitud en espacio latente
    └────────┬────────┘
              │
              ▼
    ┌─────────────────┐
    │ Gaussian Process │   Predicción + incertidumbre
    └────────┬────────┘
              │
              ▼
      μ(x), σ²(x)
```

### 2.2 Componentes Implementados

| Archivo | Descripción | Basado en Paper |
|---------|-------------|-----------------|
| `train_fsbo.py` | Entrenamiento meta-learning | Algoritmo 1 |
| `run_bo.py` | Loop de Bayesian Optimization | Algoritmo 2 |
| `fsbo_optimizer.py` | API observe/suggest | Sección 3 |
| `pipeline.py` | Integración completa | Mejoras propias |

### 2.3 Correspondencia con el Paper

| Paper | Código | Descripción |
|-------|--------|-------------|
| φ (Eq. 3) | `DeepKernelNetwork` | Red 2 capas, 128 unidades |
| k_DK (Eq. 3) | `RBFKernel + ScaleKernel` | Kernel con ARD |
| Task Aug. (Eq. 10-11) | `task_augmentation()` | Invarianza a escala |
| MLL (Eq. 5) | `ExactMarginalLogLikelihood` | Función de pérdida |
| EI | `expected_improvement()` | Acquisition function |
| Fine-tune (Sec. 3.3) | `finetune_model()` | lr=10⁻⁴, pocas epochs |
| Warm Start (Sec. 3.4) | `warm_start_model_based()` | Inicialización inteligente |

---

## 3. Datos

### 3.1 Estructura de Datos

Los datos están organizados de la siguiente manera:

```
data/
├── configspace/                    # Espacios de búsqueda
│   ├── adaboost_configspace.json
│   ├── random_forest_configspace.json
│   ├── libsvm_svc_configspace.json
│   └── autosklearn_configspace.json
│
└── representation_with_scores/     # Datos de entrenamiento
    ├── adaboost_target_representation_with_scores.csv
    ├── random_forest_target_representation_with_scores.csv
    ├── libsvm_svc_target_representation_with_scores.csv
    └── autosklearn_target_representation_with_scores.csv
```

### 3.2 Estadísticas de Datos

| Algoritmo | Muestras | Tareas | HP dims | Score medio |
|-----------|----------|--------|---------|-------------|
| AdaBoost | 4,665 | 64 | 8 | 0.744 |
| Random Forest | 10,746 | 64 | 10 | 0.747 |
| LibSVM SVC | 4,523 | 64 | 12 | 0.743 |
| AutoSklearn | 6,481 | 64 | 222 | 0.744 |

### 3.3 Generación de Datos Sintéticos

Para este proyecto académico, se generaron métricas de rendimiento sintéticas mediante `generate_synthetic_scores.py`:

- Superficie de respuesta con componentes lineales e interacciones
- Diferentes óptimos por tarea (usando hash del task_id)
- Rango realista [0.50, 0.99]
- Ruido gaussiano σ=0.03

---

## 4. Modelos Entrenados

### 4.1 Checkpoints

Se entrenaron 4 modelos FSBO con 2000 épocas cada uno:

| Modelo | Loss Final | Checkpoint |
|--------|------------|------------|
| AdaBoost | -0.0781 | `fsbo_adaboost_20260107_151935.pt` |
| Random Forest | -0.0844 | `fsbo_random_forest_20260107_151938.pt` |
| LibSVM SVC | -0.0858 | `fsbo_libsvm_svc_20260107_151942.pt` |
| AutoSklearn | -0.0851 | `fsbo_autosklearn_20260107_151946.pt` |

### 4.2 Hiperparámetros de Entrenamiento

```python
epochs = 2000
batch_size = 50
learning_rate = 1e-3
hidden_dim = 128
task_augmentation = True
```

---

## 5. Pipeline de Integración

### 5.1 Flujo Completo

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PIPELINE COMPLETO                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ENTRADA: Dataset + Algoritmos sugeridos por Meta-Learning         │
│                │                                                    │
│                ▼                                                    │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  1. EXTRACCIÓN DE META-FEATURES                             │   │
│   │     - n_samples, n_features, n_classes                      │   │
│   │     - class_imbalance, ratios                               │   │
│   │     → Usado para warm start inteligente                     │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                │                                                    │
│                ▼                                                    │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  2. ASIGNACIÓN DINÁMICA DE PRESUPUESTO                      │   │
│   │     - Más budget a algoritmos con mayor confianza           │   │
│   │     - Más budget a espacios más complejos                   │   │
│   │     → Optimiza uso de evaluaciones                          │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                │                                                    │
│                ▼                                                    │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  3. PARA CADA ALGORITMO:                                    │   │
│   │                                                             │   │
│   │     a) Warm Start Inteligente                               │   │
│   │        - Buscar configs de tareas similares (KB)            │   │
│   │        - Usar modelo pre-entrenado                          │   │
│   │                                                             │   │
│   │     b) Transfer de Hiperparámetros                          │   │
│   │        - Ponderar configs por similitud                     │   │
│   │        - Interpolar configuraciones                         │   │
│   │                                                             │   │
│   │     c) BO Loop                                              │   │
│   │        - Expected Improvement                               │   │
│   │        - Fine-tuning periódico                              │   │
│   │        - Early stopping si converge                         │   │
│   │                                                             │   │
│   │     d) Guardar en Knowledge Base                            │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                │                                                    │
│                ▼                                                    │
│   SALIDA: Mejor (algoritmo, configuración, score)                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Mejoras Implementadas

#### Mejora 1: Warm Start Inteligente con Meta-Features

```python
class IntelligentWarmStart:
    """
    Usa meta-features del dataset para encontrar tareas similares
    y transferir sus mejores configuraciones.
    """
    def get_initial_configs(self, dataset_meta, algorithm, optimizer, n_init):
        # 1. Buscar en base de conocimiento
        similar_configs = self.kb.find_similar_configs(dataset_meta, algorithm)
        
        # 2. Añadir configs transferidas (con perturbación)
        configs = [self._perturb_config(c) for c in similar_configs]
        
        # 3. Completar con sugerencias del modelo
        configs.extend(optimizer.suggest_initial(remaining))
        
        return configs
```

#### Mejora 2: Ajuste Dinámico de Presupuesto

```python
class DynamicBudgetAllocator:
    """
    Asigna más evaluaciones a:
    - Algoritmos con mayor confianza del meta-learner
    - Espacios de hiperparámetros más complejos
    """
    def allocate(self, suggestions):
        # Score = 0.6 * confianza + 0.4 * complejidad
        # Budget proporcional al score
```

#### Mejora 3: Transfer de Hiperparámetros

```python
class HyperparameterTransfer:
    """
    Transfiere conocimiento de optimizaciones anteriores.
    """
    def get_transfer_prior(self, dataset_meta, algorithm):
        # 1. Buscar tareas similares
        # 2. Ponderar por similitud y score histórico
        # 3. Retornar prior ponderado
```

### 5.3 Base de Conocimiento

El sistema mantiene una base de conocimiento que almacena:
- Meta-features de datasets procesados
- Mejores configuraciones encontradas
- Scores obtenidos

Esto permite mejorar continuamente el warm start para nuevas tareas.

---

## 6. API de Uso

### 6.1 FSBOOptimizer (observe/suggest)

```python
from fsbo_optimizer import FSBOOptimizer

# Cargar modelo pre-entrenado
optimizer = FSBOOptimizer.from_pretrained('random_forest')

# Warm start
initial_configs = optimizer.suggest_initial(n=5)
for config in initial_configs:
    score = train_and_evaluate(config)
    optimizer.observe(config, score)

# BO loop
for _ in range(budget):
    config = optimizer.suggest()
    score = train_and_evaluate(config)
    optimizer.observe(config, score)

# Resultado
best_config, best_score = optimizer.get_best()
```

### 6.2 Pipeline Completo

```python
from pipeline import run_pipeline, AlgorithmSuggestion

# Sugerencias del meta-learning
suggestions = [
    AlgorithmSuggestion('random_forest', confidence=0.85),
    AlgorithmSuggestion('adaboost', confidence=0.70),
]

# Función de evaluación
def evaluate(algorithm, config, X_tr, y_tr, X_val, y_val):
    model = get_model(algorithm, **config)
    model.fit(X_tr, y_tr)
    return model.score(X_val, y_val)

# Ejecutar pipeline
result = run_pipeline(
    X_train, y_train, X_val, y_val,
    suggested_algorithms=suggestions,
    evaluation_fn=evaluate,
    total_budget=100
)

print(f"Mejor: {result.best_algorithm} con {result.best_score:.4f}")
```

---

## 7. Resultados Experimentales

### 7.1 Test del Pipeline

```
📊 Dataset: test_synthetic (400 samples, 20 features, 3 classes)

💰 Presupuesto asignado:
   - adaboost: 20 evaluaciones (confianza=0.85)
   - random_forest: 19 evaluaciones (confianza=0.75)

🏆 Resultados:
   1. random_forest: 0.8054 (19 evals)
   2. adaboost: 0.7965 (16 evals, early stop)

⏱️ Tiempo total: 1.1 segundos
📈 Evaluaciones totales: 35
```

### 7.2 Observaciones

- **Early stopping** funcionó: AdaBoost paró en 16 de 20 evaluaciones
- **Warm start** efectivo: Scores iniciales ya en rango 0.72-0.78
- **Transfer** beneficioso: Configuraciones similares ayudaron

---

## 8. Estructura del Proyecto

```
transfer-learning/
├── data/
│   ├── configspace/              # Espacios de búsqueda (4 JSON)
│   └── representation_with_scores/  # Datos con scores (4 CSV)
├── doc/
│   ├── 2101.07667v1.pdf          # Paper FSBO
│   └── report.md                 # Este documento
├── experiments/
│   ├── checkpoints/              # Modelos entrenados (4 .pt)
│   ├── results/                  # Resultados de experimentos
│   └── knowledge_base.json       # Base de conocimiento
├── scripts/
│   ├── generate_synthetic_scores.py  # Generador de datos
│   ├── train_fsbo.py             # Entrenamiento
│   ├── run_bo.py                 # BO loop
│   ├── fsbo_optimizer.py         # API observe/suggest
│   └── pipeline.py               # Integración completa
└── requirements.txt
```

---

## 9. Referencias

- Wistuba, M., & Grabocka, J. (2021). *Few-Shot Bayesian Optimization with Deep Kernel Surrogates*. ICLR 2021.
- Wilson, A. G., et al. (2016). *Deep Kernel Learning*. AISTATS 2016.
- Snoek, J., et al. (2012). *Practical Bayesian Optimization of Machine Learning Algorithms*. NeurIPS 2012.
