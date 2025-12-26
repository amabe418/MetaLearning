# MetaLearning Project

> **Proyecto Académico**  
> Este proyecto ha sido desarrollado para la asignatura de **Machine Learning** de la carrera de **Ciencia de la Computación** de la **Facultad de Matemática y Computación** de la **Universidad de la Habana**.

Proyecto de investigación en **Meta-Learning** (Aprendizaje de Aprendizaje) que busca desarrollar y evaluar algoritmos capaces de aprender a aprender de manera eficiente.

## 📋 Descripción

Este proyecto se enfoca en el estudio y desarrollo de técnicas de meta-learning, donde los modelos aprenden a adaptarse rápidamente a nuevas tareas con pocos ejemplos (few-shot learning) o a seleccionar y configurar automáticamente algoritmos de machine learning para diferentes datasets.

## 🎯 Objetivos

- **Análisis de características de datasets**: Extraer metadatos y características relevantes de diferentes datasets para entender qué algoritmos funcionan mejor en cada contexto.
- **Predicción de rendimiento de algoritmos**: Predecir qué algoritmo de ML tendrá mejor rendimiento en un dataset nuevo basándose en características meta.
- **Selección automática de modelos**: Desarrollar sistemas que recomienden automáticamente el mejor algoritmo y sus hiperparámetros para un dataset dado.
- **Few-shot learning**: Implementar y evaluar modelos que puedan aprender nuevas tareas con pocos ejemplos.

## 📊 Fuentes de Datos

### OpenML
Utilizaremos datasets de [OpenML](https://www.openml.org/), una plataforma abierta que proporciona:
- Miles de datasets públicos con metadatos estructurados
- Resultados de experimentos de machine learning
- Características meta de datasets (número de instancias, características, clases, etc.)
- API fácil de usar para descargar datasets y metadatos

### Otras fuentes potenciales
- UCI Machine Learning Repository
- Kaggle datasets
- Datasets sintéticos generados para casos específicos

## 🛠️ Tecnologías y Herramientas

- **Python 3.8+**
- **scikit-learn**: Para algoritmos de machine learning base
- **OpenML**: Para descarga y gestión de datasets
- **pandas**: Para manipulación de datos
- **numpy**: Para operaciones numéricas
- **matplotlib/seaborn**: Para visualizaciones
- **jupyter**: Para notebooks de análisis
- **optuna/hyperopt**: Para optimización de hiperparámetros
- **meta-learn**: Librerías especializadas en meta-learning (si aplica)

## 📁 Estructura del Proyecto (versión inicial)

```
MetaLearning-/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/              # Datasets descargados de OpenML
│   ├── processed/        # Datasets preprocesados
│   └── meta_features/    # Características meta extraídas
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_meta_feature_extraction.ipynb
│   └── 03_meta_learning_experiments.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py    # Funciones para cargar datos de OpenML
│   ├── meta_features.py  # Extracción de características meta
│   ├── meta_learner.py   # Implementación de meta-learning
│   └── evaluation.py     # Métricas y evaluación
├── experiments/
│   └── results/          # Resultados de experimentos
└── docs/
    ├── state_of_the_art/  # Estado del arte 
```

## 🚀 Instalación

### Opción 1: Script Automático (Recomendado)

Ejecuta el script `run.sh` que configura automáticamente el entorno:

```bash
chmod +x run.sh
./run.sh
```

El script:
- Verifica que Python 3 esté instalado
- Crea el entorno virtual si no existe
- Instala todas las dependencias
- Ofrece opciones para ejecutar el proyecto

### Opción 2: Instalación Manual

1. Clonar el repositorio:
```bash
git clone https://github.com/tu-usuario/MetaLearning-.git
cd MetaLearning-
```

2. Crear un entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## 📝 Uso

### Descargar datasets de OpenML

```python
from src.data_loader import load_openml_dataset

# Cargar un dataset específico por ID
dataset = load_openml_dataset(dataset_id=1)

# Cargar múltiples datasets
datasets = load_openml_datasets(dataset_ids=[1, 2, 3, 4, 5])
```

### Extraer características meta

```python
from src.meta_features import extract_meta_features

meta_features = extract_meta_features(dataset)
```

### Entrenar un meta-learner

```python
from src.meta_learner import MetaLearner

meta_learner = MetaLearner()
meta_learner.train(training_datasets, training_results)
predictions = meta_learner.predict(new_dataset_meta_features)
```

## 🔬 Experimentos Planificados

1. **Análisis exploratorio de datasets OpenML**
   - Distribución de tipos de problemas (clasificación, regresión)
   - Análisis de características meta (dimensionalidad, balance de clases, etc.)

2. **Extracción de características meta**
   - Características estadísticas (media, varianza, skewness, etc.)
   - Características de información (entropía, correlación, etc.)
   - Características de complejidad (medidas de separabilidad, etc.)

3. **Meta-learning para selección de algoritmos**
   - Entrenar modelos que predigan el mejor algoritmo para un dataset
   - Comparar diferentes enfoques (landmarking, meta-features, etc.)

4. **Optimización de hiperparámetros basada en meta-learning**
   - Usar información de datasets similares para inicializar búsquedas
   - Transfer learning de configuraciones exitosas

5. **Few-shot learning**
   - Implementar modelos como MAML (Model-Agnostic Meta-Learning)
   - Evaluar en tareas de clasificación con pocos ejemplos

## 📚 Referencias

- [metalearning github](https://automl.github.io/amltk/latest/reference/metalearning/)
- [OpenML Documentation](https://docs.openml.org/)
- [Meta-Learning Survey Papers](https://arxiv.org/abs/1810.03548)
- [AutoML and Meta-Learning](https://www.automl.org/)
- [PIPES] https://github.com/cynthiamaia/PIPES/


## 👥 Autores

- Amalia Beatriz Valiente Hinojosa
- Arianne Camila Palancar Ochando
- Melani Forsythe Matos
- Jabel Resendiz Aguirre
- Jorge Alejandro Echevarría Brunet
- Noel Pérez Calvo

