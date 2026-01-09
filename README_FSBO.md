# FSBO: Few-Shot Bayesian Optimization para Optimización de Hiperparámetros

Sistema de **Transfer Learning** para optimización de hiperparámetros usando **Few-Shot Bayesian Optimization (FSBO)**.

## 🎯 Problema

La **optimización de hiperparámetros (HPO)** es costosa:
- Cada evaluación requiere entrenar un modelo completo
- Los espacios de búsqueda son grandes
- Empezar desde cero en cada nuevo dataset es ineficiente

**Solución**: Usar conocimiento de tareas previas (transfer learning) para optimizar más rápido en nuevas tareas.

## 🧠 ¿Qué es FSBO?

FSBO (Few-Shot Bayesian Optimization) es un método que:

1. **Pre-entrena** un modelo surrogate (Deep Kernel GP) en múltiples tareas
2. **Transfiere** el conocimiento a nuevas tareas
3. **Optimiza** con pocas evaluaciones gracias al conocimiento previo

**Paper**: Wistuba & Grabocka (2021) - *Few-Shot Bayesian Optimization with Deep Kernel Surrogates* (ICLR)

## 📁 Estructura del Proyecto

```
transfer-learning/
├── data/
│   ├── configspace/                    # Espacios de hiperparámetros
│   └── representation_with_scores/     # Datos con métricas
├── scripts/
│   ├── generate_synthetic_scores.py    # Generador de datos
│   ├── train_fsbo.py                   # Entrenamiento del modelo
│   ├── fsbo_optimizer.py               # API observe/suggest
│   ├── metrics.py                      # Métricas de evaluación
│   ├── baselines.py                    # Métodos de comparación
│   ├── experiments.py                  # Framework K-Fold CV
│   └── visualize.py                    # Visualizaciones
├── experiments/
│   ├── checkpoints/                    # Modelos entrenados
│   ├── results/                        # Resultados JSON
│   └── figures/                        # Gráficos
├── doc/
│   ├── technical_report.pdf            # Documentación técnica
│   └── experimental_report.pdf         # Resultados experimentales
└── requirements.txt
```

## 🚀 Instalación

```bash
# Clonar repositorio
git clone https://github.com/usuario/MetaLearning-.git
cd MetaLearning-/transfer-learning

# Crear entorno virtual (opcional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o: venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt
```

## 📦 Dependencias

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
scipy>=1.7.0
torch>=1.10.0
gpytorch>=1.6.0
tqdm>=4.62.0
matplotlib>=3.5.0
```

## 💻 Uso

### 1. Entrenar el modelo FSBO

```bash
# Entrenar para un algoritmo específico
python scripts/train_fsbo.py --algorithm adaboost --epochs 2000

# Entrenar para todos los algoritmos
python scripts/train_fsbo.py --algorithm all
```

### 2. Usar el optimizador (API observe/suggest)

```python
from fsbo_optimizer import FSBOOptimizer

# Cargar modelo pre-entrenado
optimizer = FSBOOptimizer.from_pretrained('random_forest')

# Warm start: configuraciones iniciales prometedoras
initial_configs = optimizer.suggest_initial(n=5)
for config in initial_configs:
    score = train_and_evaluate(model, config)
    optimizer.observe(config, score)

# Loop de optimización
for _ in range(25):
    config = optimizer.suggest()           # Sugerir siguiente config
    score = train_and_evaluate(model, config)
    optimizer.observe(config, score)       # Registrar resultado

# Obtener mejor configuración
best_config, best_score = optimizer.get_best()
```

### 3. Ejecutar experimentos

```bash
# Experimento completo con K-Fold CV
python scripts/experiments.py \
    --algorithm all \
    --k_folds 5 \
    --n_trials 30 \
    --n_seeds 3 \
    --methods fsbo random gp-rs

# Generar visualizaciones
python scripts/visualize.py \
    --results experiments/results/ \
    --output experiments/figures/
```

## 📊 Resultados

FSBO supera consistentemente a los baselines en todos los algoritmos evaluados:

| Algoritmo | FSBO (NR↓) | Random | GP-RS |
|-----------|------------|--------|-------|
| AdaBoost | **0.189** | 0.195 | 0.197 |
| Random Forest | **0.230** | 0.253 | 0.259 |
| LibSVM_SVC | **0.196** | 0.217 | 0.200 |
| AutoSklearn | **0.332** | 0.341 | 0.334 |

*NR = Normalized Regret (menor es mejor)*

## 🔗 Integración con Meta-Learning

Este módulo está diseñado para integrarse con el componente de meta-learning:

```python
from fsbo_optimizer import optimize_algorithms

# Meta-learning sugiere algoritmos para el dataset
suggested_algorithms = meta_learner.suggest(X, y)
# -> ['random_forest', 'adaboost']

# FSBO optimiza hiperparámetros de cada uno
results = optimize_algorithms(
    algorithms=suggested_algorithms,
    evaluation_fn=lambda alg, hp: train_evaluate(X, y, alg, hp),
    budget_per_algorithm=30
)

# Mejor combinación (algoritmo + hiperparámetros)
best_alg = max(results, key=lambda a: results[a].best_score)
print(f"Mejor: {best_alg} con {results[best_alg].best_config}")
```

## 📚 Documentación

- **[technical_report.pdf](doc/technical_report.pdf)**: Documentación técnica completa (17 páginas)
- **[experimental_report.pdf](doc/experimental_report.pdf)**: Análisis de resultados experimentales
- **[EXPERIMENTS.md](doc/EXPERIMENTS.md)**: Guía del framework de experimentación

## 🧪 Algoritmos Soportados

- AdaBoost
- Random Forest
- LibSVM SVC
- AutoSklearn

## 📖 Referencias

```bibtex
@inproceedings{wistuba2021fsbo,
  title={Few-Shot Bayesian Optimization with Deep Kernel Surrogates},
  author={Wistuba, Martin and Grabocka, Josif},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```



Proyecto académico - MetaLearning

---

**Fecha**: Enero 2026

