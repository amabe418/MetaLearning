# Survey – Meta-Learning

# *Paper 1: Meta-Learning: A Survey (Joaquin Vanschoren et al., 2017)
**Link:** [Meta-Learning: A Survey](state_of_the_art/metaLearning.pdf)
**Idea principal:** explica ideas iniciales y referencias de varios autores de las diferentes formas de hacer metalearning

---

# Paper 2: Experiment databases (Vanschoren · Blockeel · Pfahringer · Holmes 2012)
**Link:** [Experiment dataset](state_of_the_art/Experiment_databases.pdf)
**Idea principal:** aprendizaje de parámetros iniciales que permiten adaptación rápida.  
**Contribuciones clave:**
- Meta-learning basado en gradientes.
- Uso de inner/outer loops.
- SOTA en few-shot classification.

**Limitaciones:**
- Costoso computacionalmente.
- Inner loop inestable si el learning rate no es adecuado.

---

# Paper 3: Alpha D3M: Machine Learning Pipeline Synthsis

## 📌 Contexto en el estado del arte

AlphaD3M se sitúa en la intersección de:

* **Meta-learning**
* **AutoML**
* **Síntesis automática de pipelines**
* **Reinforcement Learning profundo (AlphaZero-like)**

Pertenece a la línea de investigación impulsada por **DARPA D3M**, cuyo objetivo es:

> Resolver *cualquier tarea ML* sobre *cualquier dataset*, sintetizando automáticamente pipelines completos, **explicables y eficientes**.

---

## 🎯 ¿De qué trata el paper?

El paper introduce **AlphaD3M**, un sistema AutoML que:

* Modela la **construcción de pipelines ML como un juego de un solo jugador**
* Usa:

  * **Redes neuronales recurrentes (LSTM)**
  * **Monte Carlo Tree Search (MCTS)**
  * **Auto-juego (self-play)** al estilo **AlphaZero**
* Aprende a **editar pipelines** mediante:

  * inserción
  * eliminación
  * reemplazo de componentes

👉 El resultado es un sistema:

* competitivo con AutoSklearn, TPOT y Autostacker
* **mucho más rápido**
* **explicable por diseño**

---

## 🧩 Idea central (contribución conceptual)

### 🔑 Reformulación clave

> **La síntesis de pipelines es un problema de búsqueda secuencial**, no solo de optimización de hiperparámetros.

Se modela como:

| Concepto   | AlphaZero    | AlphaD3M                            |
| ---------- | ------------ | ----------------------------------- |
| Juego      | Ajedrez / Go | AutoML                              |
| Estado     | Tablero      | (Dataset + tarea + pipeline actual) |
| Acción     | Movimiento   | Editar pipeline                     |
| Recompensa | Ganar        | Performance del pipeline            |

---

## 🧠 ¿Qué parte del *meta-learning* explica?

AlphaD3M **NO se centra** en:

* selección manual de meta-features
* ranking clásico de algoritmos
* kNN meta-learning

👉 Se centra en **meta-learning implícito**, aprendido por la red.

### Meta-learning en AlphaD3M =

> aprender **patrones recurrentes de pipelines efectivos** a través de múltiples datasets y tareas.

### Representación del estado (meta-learning)

El estado incluye:

1. **Meta-data del dataset**
   (no detallado exhaustivamente en el paper)
2. **Definición de la tarea**
3. **Pipeline completo actual**
4. **Historial implícito de decisiones**

⚠️ Importante:

* **NO lista explícitamente las meta-features**
* El *aprendizaje* ocurre dentro de la red neuronal

---

## 🧪 ¿Qué datos y benchmarks usa?

* **313 datasets tabulares**

  * **296 de OpenML**
* Tareas:

  * Clasificación binaria
  * Clasificación multiclase
  * Regresión
* NO se listan explícitamente:

  * `dataset_id`
  * `task_id`
  * suites OpenML

👉 El paper **NO está orientado a reproducibilidad fina**, sino a demostrar el enfoque.

---

## ⚙️ ¿Qué algoritmos/pipelines usa?

* Pipelines compuestos por:

  * Preprocesamiento
  * Feature extraction
  * Feature selection
  * Estimadores
  * Post-procesamiento
* Baseline:

  * SGD (clasificación y regresión)
* Los algoritmos concretos **no se enumeran formalmente**
* Se trabaja con **primitives** (concepto D3M), no con “algoritmos aislados”

---

## 🧠 ¿Cómo funciona el sistema (pipeline conceptual)?

```text
Dataset + Task
      ↓
Estado inicial (pipeline base)
      ↓
Red neuronal (LSTM)
  → predice:
    - probabilidad de acciones
    - performance estimada
      ↓
MCTS
  → explora ediciones de pipeline
      ↓
Evaluación real del pipeline
      ↓
Reward (performance)
      ↓
Entrenamiento (self-play)
```

La red aprende **qué editar, cuándo y por qué**.

---

## 📈 Resultados principales

* AlphaD3M:

  * supera a pipelines base en ~75% de datasets
  * es **comparable en performance** con AutoSklearn, TPOT y Autostacker
  * es **~10× más rápido**
* Tiempo:

  * horas → minutos
* Ventaja clave:

  * **explicabilidad estructural** (ediciones del pipeline)

---

## 🧠 Aportes principales del paper

### ✔️ Conceptuales

* Primera formulación **AlphaZero-like** para AutoML
* Pipeline synthesis como **juego secuencial**
* Meta-learning **end-to-end**, no basado en meta-features manuales

### ✔️ Técnicos

* Uso combinado de:

  * LSTM
  * MCTS
  * Self-play
* Ediciones de pipeline como acciones explicables

### ✔️ Prácticos

* Más rápido que AutoML clásico
* Escalable a espacios enormes de pipelines

---

## ❌ Qué NO explica (y por qué)

| Elemento                 | ¿Está? | Razón                       |
| ------------------------ | ------ | --------------------------- |
| Meta-features explícitas | ❌      | Aprendidas implícitamente   |
| IDs OpenML               | ❌      | No es paper de benchmarking |
| Ranking de algoritmos    | ❌      | Trabaja a nivel pipeline    |
| Reproducibilidad exacta  | ❌      | Enfoque conceptual          |

Esto es **intencional**, no un fallo.

---

## 🔗 Código e implementaciones

### ✔️ Implementación asociada (D3M / AlphaD3M)

* Proyecto D3M (DARPA):

  * [https://github.com/VIDA-NYU/d3m](https://github.com/VIDA-NYU/d3m)
* Componentes relacionados con AlphaD3M:

  * [https://github.com/VIDA-NYU/alphad3m](https://github.com/VIDA-NYU/alphad3m)

⚠️ Nota:

* El código es **complejo**
* Usa primitives D3M
* No es tan “plug-and-play” como AutoSklearn

---

## 🧠 Relación con otros enfoques de meta-learning

| Enfoque               | Ejemplo         | AlphaD3M |
| --------------------- | --------------- | -------- |
| Meta-features + kNN   | Brazdil, Soares | ❌        |
| Ranking de algoritmos | OpenML          | ❌        |
| AutoML Bayesian       | AutoSklearn     | Parcial  |
| Evolutivo             | TPOT            | Parcial  |
| RL + search           | ❌               | ✅        |

---

## 📝 Resumen corto (para sección *Related Work*)

> *AlphaD3M frames AutoML pipeline synthesis as a single-player sequential decision-making problem inspired by AlphaZero. Instead of relying on explicit meta-features or algorithm rankings, it learns an implicit meta-representation of datasets and tasks through self-play, using neural networks and Monte-Carlo Tree Search to iteratively edit pipelines. This approach achieves competitive performance while being significantly faster and explainable by design.*

---

# *Paper4 : Automatic Exploration of Machine Learning Experiments on OpenML

**Link:** [Automatic Exploration of Machine Learning Experiments on OpenML](state_of_the_art/Automatic_Exploration_of_Machine_Learning_Experiments_on_OpenML.pdf)

**Daniel Kühn, Philipp Probst, Janek Thomas, Bernd Bischl**

---

## 📌 Contexto en el estado del arte

Este paper se sitúa en la línea de:

* **Meta-learning basado en experiencias**
* **Análisis empírico de hiperparámetros**
* **Benchmarking masivo y reproducible**
* **OpenML como infraestructura científica**

Es un paper **fundacional** para:

* meta-learning moderno,
* AutoML,
* y análisis de *hyperparameter importance*.

👉 A diferencia de AlphaD3M, **NO propone un nuevo algoritmo**, sino que crea **infraestructura experimental a gran escala**.

---

## 🎯 ¿De qué trata el paper?

El paper introduce un **meta-dataset masivo** construido automáticamente que:

* ejecuta **millones de experimentos ML**
* sobre **datasets reales de OpenML**
* con **muestreo aleatorio de hiperparámetros**
* de forma **totalmente automática** mediante el **OpenML Random Bot**

El objetivo central es:

> Entender empíricamente cómo los hiperparámetros influyen en el rendimiento de los algoritmos.

---

## 🧩 Idea central

### 🔑 Contribución clave

> **La comunidad necesita grandes bases de datos experimentales para estudiar ML empíricamente**, no solo benchmarks pequeños.

Este trabajo:

* genera esa base de datos
* la publica
* la integra con OpenML

---

## 🧪 ¿Qué datos usa?

### ✔️ Datasets

* **38 datasets de OpenML**
* Clasificación supervisada
* Datasets públicos, variados y reales

⚠️ Limitación importante:

* **38 datasets ≠ diversidad extrema**
* pero **muchísimos experimentos por dataset**

---

## ⚙️ ¿Qué algoritmos evalúa?

Evalúa **6 algoritmos clásicos**, elegidos por:

* estabilidad
* popularidad
* interpretabilidad

Típicamente (según el paper y contexto OpenML):

* Random Forest
* Support Vector Machines
* k-Nearest Neighbors
* Decision Trees
* Naive Bayes
* Logistic Regression

👉 **Aquí sí hay algoritmos explícitos**, aunque el foco no es compararlos, sino **estudiar su espacio de hiperparámetros**.

---

## 🔧 ¿Cómo se generan los experimentos?

### 🔁 OpenML Random Bot

Un *bot automático* que:

1. Selecciona un dataset OpenML
2. Selecciona un algoritmo
3. Muestra aleatoriamente hiperparámetros
4. Ejecuta validación cruzada
5. Sube los resultados a OpenML

### Escala:

* Hasta **20.000 configuraciones por algoritmo y dataset**
* ≈ **2.5 millones de runs**
* Cada run:

  * algoritmo
  * hiperparámetros
  * score
  * tiempo
  * dataset_id
  * task_id

👉 Todo queda **versionado y reproducible en OpenML**.

---

## 🧠 ¿Qué parte del meta-learning aborda?

Este paper es **meta-learning de nivel bajo (experimental)**.

### ✔️ Lo que SÍ cubre:

* Construcción de una **experience database**
* Relación:

  ```
  (dataset, algoritmo, hiperparámetros) → performance
  ```
* Base para:

  * algorithm selection
  * hyperparameter importance
  * surrogate models
  * AutoML

### ❌ Lo que NO cubre:

* No entrena meta-modelos
* No propone ranking de algoritmos
* No hace recomendaciones directamente

👉 Es **infraestructura**, no el meta-learner.

---

## 🧠 Meta-features

⚠️ Punto importante (y frecuente confusión):

* **NO calcula meta-features del dataset**
* **NO las necesita**

Porque el objetivo es:

> estudiar el *response surface* de hiperparámetros, no predecir entre datasets.

---

## 📈 Resultados y análisis

El paper muestra:

* distribuciones de rendimiento
* sensibilidad a hiperparámetros
* regiones estables vs inestables
* interacción entre hiperparámetros

👉 Conclusión clave:

> El rendimiento depende fuertemente de pocas configuraciones bien elegidas, justificando AutoML.

---

## 🔗 Código y datos

### ✔️ Datos

* Todos los resultados están en **OpenML**
* Accesibles vía:

  * OpenML API
  * `openml-python`
  * runs históricos

### ✔️ Código

* Scripts del Random Bot (históricos)
* Infraestructura OpenML

No hay un “repo bonito”, pero **los datos están completamente disponibles**.

---

## 🧠 Relación con otros trabajos

| Trabajo        | Rol                       |
| -------------- | ------------------------- |
| Brazdil et al. | Meta-learning clásico     |
| AlphaD3M       | Síntesis de pipelines     |
| AutoSklearn    | Optimización              |
| **Este paper** | Base experimental         |
| PIPES          | Meta-dataset de pipelines |

👉 Este paper es **la capa base** sobre la que se apoyan muchos otros.

---

## 📝 Resumen corto (para *Related Work*)

> *Kühn et al. present a large-scale experimental meta-dataset generated via automated random sampling of hyperparameters on OpenML. Covering millions of runs across multiple datasets and algorithms, this work provides the empirical foundation required for studying hyperparameter effects, algorithm behavior, and for training meta-learning and AutoML systems, rather than proposing a new meta-learning method itself.*

---

## 🧠 Diferencia clave con AlphaD3M

| AlphaD3M            | Este paper      |
| ------------------- | --------------- |
| Método nuevo        | Infraestructura |
| RL + MCTS           | Random sampling |
| Pipelines completos | Algoritmos + HP |
| Implícito           | Explícito       |
| End-to-end          | Base de datos   |

---

## 🎯 Para TU proyecto

Este paper es **perfecto** para justificar:

* uso de OpenML
* necesidad de meta-datasets grandes
* análisis empírico
* reproducibilidad

Y combina muy bien con:

* Brazdil (ranking)
* PIPES (pipelines)
* AlphaD3M (síntesis)

---

# **Paper5 : 🧠 On the Predictive Power of Meta-Features in OpenML

**Link:** [ On the Predictive Power of Meta-Features in OpenML](state_of_the_art/Bilalli%20et%20al.pdf)

**Besim Bilalli, Alberto Abelló, Tomàs Aluja-Banet (UPC, BarcelonaTech)**

---

## 📌 Contexto en el estado del arte

* Este paper se sitúa en la línea de **meta-learning basado en meta-features**.
* Problema central:

  > Selección automática de algoritmos (model/algorithm selection) depende de la **caracterización del dataset** mediante meta-features.
* Destinado a **asistir usuarios no expertos** en la selección de modelos.

A diferencia de AlphaD3M, aquí **no se construyen pipelines ni se usa RL**; el enfoque es **analítico y estadístico**, centrado en **meta-features predictivas**.

---

## 🎯 ¿De qué trata el paper?

* Analiza la **capacidad predictiva de diferentes meta-features** en OpenML.
* Usa **factor analysis** para:

  1. Extraer **latent features** (agrupaciones de meta-features con características comunes)
  2. Evaluar su relación con el rendimiento de 4 algoritmos de clasificación en cientos de datasets
  3. Seleccionar las **latent features más predictivas**
* Finalmente, realiza **meta-learning** usando las latent features seleccionadas para mejorar la recomendación de algoritmos.

---

## 🧩 Idea central

> **Mejorar la efectividad del meta-learning** mediante la identificación de las meta-features más predictivas, usando análisis estadístico en datasets de OpenML.

* La aproximación combina:

  * **Feature extraction**: factor analysis → latent features
  * **Feature selection**: elegir las más predictivas
  * **Meta-learning**: usar esas features para predecir el rendimiento de algoritmos

---

## 🧪 Datos y experimentos

* **Datasets**: cientos de datasets públicos en OpenML
* **Algoritmos evaluados**: 4 algoritmos de clasificación (no se listan todos explícitamente)
* **Evaluación**: relación entre latent features y 3 métricas de desempeño (accuracy, f1, etc.)
* **Resultado**: selección de latent features con alto poder predictivo

---

## 🔧 Qué parte del meta-learning aborda

### ✔️ Lo que sí hace:

* Meta-learning basado en **caracterización del dataset**
* Extracción de **features latentes predictivas**
* Mejora de la **predicción del algoritmo óptimo**

### ❌ Lo que NO hace:

* No construye pipelines completos
* No usa RL ni search-based AutoML
* No genera ranking directo en OpenML de forma exhaustiva

---

## 🧠 Contribuciones principales

### ✔️ Conceptuales

* Demuestra que **la elección de meta-features es crítica** para meta-learning
* Introduce **latent features** como representación compacta y predictiva

### ✔️ Técnicas

* Uso de **factor analysis** para agrupar meta-features
* Selección basada en relación estadística con rendimiento de algoritmos

### ✔️ Prácticos

* Diseña una **aplicación para recuperar meta-datos de OpenML**
* Mejora procesos de **algorithm recommendation** para usuarios no expertos

---

## 📈 Resultados clave

* Algunas latent features explican gran parte de la variabilidad en performance
* Meta-learning con features seleccionadas **mejora la recomendación de algoritmos**
* Validación empírica en **hundreds of OpenML datasets**

---

## 🔗 Implementación / Datos

* **Datos y meta-features** disponibles en OpenML
* Aplicación para **extraer meta-data**: facilita replicación de experiments
* Código específico **no publicado** como repo, pero los datos son accesibles

---

## 🧠 Comparación con otros trabajos

| Trabajo            | Rol                                  |
| ------------------ | ------------------------------------ |
| Brazdil et al.     | Meta-learning clásico (kNN, ranking) |
| AlphaD3M           | Pipeline synthesis y RL              |
| Kühn et al.        | Random hyperparameter experiments    |
| **Bilalli et al.** | Meta-feature selection y prediction  |

👉 Este paper es **una pieza clave para elegir qué features usar** antes de entrenar un meta-learner.

---

## 📝 Resumen corto (para *Related Work*)

> *Bilalli et al. study the predictive power of meta-features in OpenML. By extracting latent features through factor analysis and selecting the most predictive ones, they demonstrate improved meta-learning performance in algorithm recommendation. Unlike pipeline-synthesis approaches, this work focuses on understanding dataset characterizations and how they relate to algorithm performance.*

---

# *Paper6 : 🧠 Characterizing the Applicability of Classification Algorithms Using Meta-Level Learning

**Link:** [Characterizing the Applicability of Classification Algorithms Using Meta-Level Learning](state_of_the_art/characterizing-the-applicability-of-classification-4gmkiy2ggj.pdf)

**Pavel Brazdil, João Gama, Bob Henery**

---

## 📌 Contexto en el estado del arte

* Este paper es un **clásico del meta-learning** aplicado a la **selección de algoritmos de clasificación**.

* Problema central:

  > Dado un nuevo dataset, ¿qué algoritmos de clasificación son más adecuados?

* La idea es usar **información previa (meta-level)** sobre datasets y algoritmos para **generar recomendaciones automáticas**.

---

## 🎯 ¿De qué trata el paper?

* Realiza un **estudio comparativo** de distintos algoritmos: machine learning, estadísticos y redes neuronales.

* Utiliza **meta-level learning**, es decir:

  1. Caracteriza datasets mediante **medidas estadísticas e información teórica**.
  2. Combina estas características con los resultados de tests previos.
  3. Entrena un **sistema de meta-learning** para predecir qué algoritmos son más adecuados para un dataset dado.

* El sistema genera **reglas automáticas**, que incluso pueden ser editadas por un usuario.

---

## 🧩 Idea central

> Usar resultados empíricos de algoritmos previos + características de datasets para entrenar un meta-modelo que recomiende algoritmos de clasificación adecuados para nuevos datasets.

* **Meta-features utilizadas**: estadísticas, medidas de información, propiedades de la distribución de datos.
* **Meta-modelo**: machine learning aplicado sobre meta-datos, generando reglas y puntuaciones informativas.

---

## 🧪 Datos y experimentos

* Datasets diversos (no especifica todos, típico de papers clásicos de Brazdil).
* Algoritmos evaluados: varios clasificadores clásicos, incluyendo ML, estadísticos y redes neuronales.
* Evaluación:

  * Los datasets se caracterizan mediante medidas estadísticas y de información.
  * El sistema aprende la relación entre estas características y el desempeño de los algoritmos.
  * Se generan recomendaciones para datasets nuevos con un **information score**.

---

## 🔧 Qué parte del meta-learning aborda

### ✔️ Lo que sí hace:

* **Algorithm selection** basado en meta-learning.
* **Construcción de meta-features** (dataset characterization).
* **Generación de reglas** que explican la recomendación de algoritmos.

### ❌ Lo que NO hace:

* No realiza síntesis de pipelines completos.
* No usa RL ni AutoML moderno.
* No trabaja con millones de runs como OpenML Random Bot.

---

## 🧠 Contribuciones principales

### ✔️ Conceptuales

* Introduce el concepto de **meta-level learning** aplicado a la selección de algoritmos.
* Muestra cómo combinar **características del dataset + desempeño previo** para predecir la idoneidad de algoritmos.

### ✔️ Técnicas

* Medidas estadísticas e información teórica como meta-features.
* Sistema de **reglas automáticas** generadas por ML.

### ✔️ Prácticos

* Herramienta para recomendar algoritmos en datasets nuevos.
* Mejora la experiencia de usuarios no expertos.

---

## 📈 Resultados clave

* El sistema puede **predecir qué algoritmos son más adecuados** para un dataset nuevo.
* Los scores de información permiten **clasificar y priorizar algoritmos**.
* Experimentos muestran que **las recomendaciones son útiles y viables** en la práctica.

---

## 🔗 Implementación / Datos

* No hay repo publicado moderno.
* Experimentos y meta-features disponibles en el paper y referencias históricas de Brazdil.
* Idea replicable usando datasets de OpenML y características estadísticas.

---

## 🧠 Relación con otros trabajos

| Trabajo                 | Rol                                              |
| ----------------------- | ------------------------------------------------ |
| **Brazdil et al. 2003** | Meta-learning clásico de selección de algoritmos |
| Bilalli et al.          | Meta-feature extraction y predicción             |
| AlphaD3M                | Síntesis de pipelines y AutoML                   |
| Kühn et al.             | Dataset-algorithm meta-dataset                   |

> Este paper es un **punto de partida histórico** para meta-learning basado en dataset characterization y algorithm recommendation.

---

## 📝 Resumen corto (para *Related Work*)

> *Brazdil et al. present a meta-learning approach to algorithm selection. Using dataset characterization through statistical and information-theoretic measures, combined with previous algorithm performance, the system generates rules and information scores to recommend suitable classifiers for new datasets. This work lays the foundation for meta-feature based algorithm recommendation in modern AutoML pipelines.*

---

# **Paper 7: 🧠 Experiment Databases (Vanschoren et al., 2009/2011)

**Link:** [🧠 Experiment Databases (Vanschoren et al., 2009/2011)](state_of_the_art/Experiment_databases.pdf)


## 📌 Contexto

* Muchos papers de ML generan resultados experimentales, pero **gran parte de los detalles se pierden** tras la publicación.

* Esto dificulta:

  * Reproducibilidad
  * Reutilización de experimentos para nuevos estudios
  * Comparaciones sistemáticas entre algoritmos y datasets

* Solución propuesta: **bases de datos de experimentos** (Experiment Databases) que almacenan:

  * Datasets
  * Algoritmos
  * Hiperparámetros
  * Resultados de evaluación

---

## 🎯 ¿De qué trata el paper?

* Presenta un **framework colaborativo** para almacenar y compartir resultados experimentales de ML.

* Organiza experimentos automáticamente en **bases de datos públicas**, permitiendo:

  * Reutilizar experimentos previos
  * Analizar resultados a gran escala
  * Responder preguntas de investigación sobre algoritmos, hiperparámetros y datasets

* Actualmente contiene **más de 650,000 experimentos de clasificación**.

---

## 🧩 Relevancia para tu proyecto

Si tu objetivo es **meta-learning con metafeatures** para recomendar algoritmos, hiperparámetros o pipelines:

1. **Base de datos de experimentos = fuente de meta-datos**

   * Cada registro contiene:

     * Dataset (con sus características/metafeatures)
     * Algoritmo usado
     * Hiperparámetros
     * Resultado obtenido (accuracy, F1, etc.)

2. **Permite entrenar un meta-learner**

   * Puedes usar los metadatos para predecir:

     * Qué algoritmo funcionará mejor en un dataset nuevo
     * Qué combinación de hiperparámetros es prometedora
     * Incluso qué pipeline sería adecuado si combinas algoritmos y preprocesamiento

3. **Facilita reproducibilidad y comparaciones**

   * Puedes validar tus recomendaciones comparando con resultados ya almacenados

---

## 🔧 Qué NO hace este paper

* No propone **algoritmos de AutoML**
* No construye pipelines automáticamente
* No aplica directamente meta-learning, aunque **los datos que organiza son perfectos para hacerlo**

---

## ✅ Conclusión para tu proyecto

Tiene el estilo de **PIPES** y de **OpenML**, solo que con matices:

* **OpenML**:

  * Plataforma actual, online y colaborativa.
  * Permite **almacenar datasets, runs de algoritmos, metafeatures, experimentos completos**.
  * Facilita **descargar datasets, resultados y metafeatures para meta-learning o AutoML**.

* **PIPES**

  * Es un **framework de experimentación y evaluación de pipelines**, más enfocado en **evaluar algoritmos y generar recomendaciones** basadas en metafeatures.
  * Incluye **algoritmos evaluados, rankings y pipelines sugeridos**, más cercano a un **sistema de recomendación**.

* **Experiment Databases (Vanschoren et al., 2009)**:

  * Es **el antecesor conceptual de OpenML**.
  * Base de datos de experimentos para **guardar, organizar y compartir resultados de ML**.
  * No ejecuta pipelines ni hace recomendaciones automáticas, pero **los datos que contiene permiten entrenar meta-learners o sistemas de recomendación**.

---

💡 **Resumen comparativo simple**:

| Sistema / Base       | Qué hace                                                          | Meta-learning útil para tu proyecto?                            |
| -------------------- | ----------------------------------------------------------------- | --------------------------------------------------------------- |
| OpenML               | Almacena datasets, runs, metafeatures; API para acceso y descarga | ✅ Directamente, listo para entrenar meta-learners               |
| PIPES                | Evalúa pipelines y algoritmos; genera rankings                    | ✅ Sí, más cercano a recomendación automática                    |
| Experiment Databases | Solo guarda y organiza resultados experimentales                  | ✅ Sí, pero necesitas construir tu meta-learner sobre esos datos |

---

# **Paper8: 🧠 Fast Algorithm Selection using Learning Curves

**Link:** [🧠 Fast Algorithm Selection using Learning Curves](state_of_the_art/Fast_Learning_curve.pdf)

**Jan N. van Rijn, Salisu Mamman Abdulrahman, Pavel Brazdil, Joaquin Vanschoren**

---

## 📌 Contexto en el estado del arte

* Problema central:

  > Encontrar un clasificador y su configuración de hiperparámetros que funcionen bien en un dataset dado.

* Dificultad: Evaluar todas las combinaciones posibles **toma demasiado tiempo**.

* Solución: Predecir qué algoritmos son más prometedores a partir de **pequeñas muestras de datos**.

---

## 🎯 Idea central del paper

* El objetivo es **rankear algoritmos en lugar de clasificarlos**:

  * La primera recomendación no siempre es la mejor
  * Se generan **múltiples recomendaciones** basadas en desempeño medido sobre pequeñas muestras

* Introduce el concepto de **Loss-Time Curves**:

  * Visualizan **cuánto tiempo (budget)** se necesita para llegar a una solución aceptable
  * Permite evaluar rankings de algoritmos considerando **tiempo y rendimiento**

* El método propuesto:

  1. Toma pequeñas muestras de un dataset.
  2. Evalúa rápidamente los clasificadores.
  3. Genera un ranking adaptado para minimizar tiempo y maximizar precisión.

---

## 🧩 Datos y experimentos

* Se usan **datasets de benchmark** (probablemente OpenML, aunque no especifica todos).
* Clasificadores evaluados: varios **algoritmos clásicos de ML**.
* Resultados:

  * El método converge **muy rápido** a soluciones aceptables
  * Permite comparar rankings de algoritmos considerando **tiempo de entrenamiento y precisión**

---

## 🔧 Qué parte del meta-learning aborda

### ✔️ Lo que hace:

* **Algorithm selection** basado en rendimiento medido sobre pequeñas muestras.
* **Ranking de algoritmos**, no solo predicción del mejor.
* Considera **trade-off entre tiempo y precisión**.

### ❌ Lo que NO hace:

* No hace síntesis completa de pipelines (solo selecciona algoritmos).
* No extrae nuevas metafeatures, sino que usa características existentes de datasets.

---

## ✅ Contribuciones principales

### ✔️ Conceptuales

* Introduce **evaluación de rankings de algoritmos con Loss-Time Curves**.
* Propone un **meta-approach rápido** para selección de algoritmos basado en subsampling.

### ✔️ Técnicas

* Mide rendimiento de clasificadores en **subsets del dataset**.
* Genera **ranking adaptado a tiempo y precisión**.

### ✔️ Prácticos

* Permite **seleccionar algoritmos prometedores rápidamente**.
* Útil para **AutoML y meta-learning** cuando evaluar todos los algoritmos es costoso.

---

## 🔗 Relevancia para el proyecto

* Muy relevante para **meta-learning basado en metafeatures y selección de algoritmos**.
* Conceptos aplicables:

  * Usar **pequeñas muestras** para predecir rendimiento de algoritmos
  * Generar **ranking de algoritmos** en lugar de solo seleccionar uno
  * Considerar **tiempo como parte del criterio** de selección

---

# *Paper9:🧠 PIPES: A Meta-dataset of Machine Learning Pipelines

**Link:** [🧠 PIPES: A Meta-dataset of Machine Learning Pipelines](state_of_the_art/PIPES:a_meta-dataset_of_ML.pdf)

**Cynthia Moreira Maia, Lucas B. V. de Amorim, George D. C. Cavalcanti, Rafael M. O. Cruz**

---

## 📌 Contexto

* Problema central:

  > Los sistemas de meta-learning requieren datasets de referencia con información sobre algoritmos, hiperparámetros y pipelines completos para entrenar meta-learners.

* Dificultad:

  * Los datasets de meta-learning existentes **no contienen pipelines completos ni metafeatures detalladas**.
  * La evaluación de pipelines sobre muchos datasets es **costosa y poco reproducible**.

* Solución: **PIPES**, un meta-dataset que organiza información sobre pipelines de ML y sus resultados, optimizado para meta-learning y recomendación automática.

---

## 🎯 Idea central

* PIPES es un **meta-dataset de pipelines de ML** que contiene:

  * Datasets tabulares de benchmark
  * Metafeatures de los datasets
  * Algoritmos usados en pipelines (clasificación y regresión)
  * Hiperparámetros
  * Rendimiento de cada pipeline (accuracy, RMSE, etc.)

* Incluye **información estructurada** para facilitar:

  * Entrenamiento de meta-learners
  * Predicción del rendimiento de pipelines
  * Recomendación de algoritmos y configuraciones

* También proporciona una **API** para acceder a los datos y realizar consultas sobre datasets, algoritmos y metafeatures.

---

## 🧩 Datos y experimentos

* Contiene **pipelines completos**, desde preprocesamiento hasta estimadores.
* Pipelines evaluados sobre **varios datasets tabulares** (principalmente de OpenML).
* Permite análisis de **eficiencia, reproducibilidad y generalización de pipelines**.

---

## 🔧 Qué parte del meta-learning aborda

### ✔️ Lo que hace:

* **Recomendación de pipelines y algoritmos** basada en metafeatures.
* Facilita **aprendizaje meta sobre rendimiento histórico** de algoritmos y configuraciones.
* Permite construir **modelos predictivos sobre qué pipeline funcionará mejor** en un dataset nuevo.

### ❌ Lo que NO hace:

* No genera pipelines automáticamente; **proporciona datos para entrenar un meta-learner que lo haga**.
* No propone nuevas arquitecturas de aprendizaje; **es una base de datos/meta-dataset**.

---

## ✅ Contribuciones principales

1. **Meta-dataset completo de pipelines** para clasificación y regresión.
2. Incluye **metafeatures, algoritmos, hiperparámetros y resultados** para cada dataset.
3. Proporciona **API y estructura reproducible** para experimentos de meta-learning.
4. Facilita **evaluación de recomendaciones y análisis de algoritmos/pipelines**.

---

## 🔗 Relevancia para tu proyecto

* Muy útil si quieres entrenar un **meta-learner que prediga el mejor pipeline o algoritmo** para un dataset nuevo usando metafeatures.
* Combina lo mejor de **OpenML (datasets y metafeatures)** con la **evaluación histórica de pipelines**, similar a lo que proponía **Experiment Databases** pero más estructurado para pipelines completos.
* Permite **reproducibilidad y benchmarking** de algoritmos y configuraciones.

---

# **Paper10: 🧠 Pairwise Meta‑Rules for Better Meta‑Learning‑Based Algorithm Ranking

**Link:** [🧠 Pairwise Meta‑Rules for Better Meta‑Learning‑Based Algorithm Ranking](state_of_the_art/Quan_Sun.pdf)

**Quan Sun · Bernhard Pfahringer**

---

## 📌 Contexto general

* El problema central del paper es **algorithm selection / ranking**:

  > Dado un nuevo dataset, ¿qué algoritmos probablemente funcionarán mejor?

* En meta‑learning, esto suele abordarse aprendiendo de **experiencias previas** (datasets + rendimiento de algoritmos) para predecir rankings en datasets nuevos.

* El **objetivo específico** de este trabajo es mejorar cómo se construyen esos rankings usando **reglas meta** más robustas, basadas en comparaciones **par a par** entre algoritmos.

---

## 🎯 Idea central del paper

### 🤔 Problema al que responde

* Muchos enfoques de meta‑learning intentan predecir **el mejor algoritmo** directamente o producen un **ranking global** de algoritmos.
* Sin embargo, esos enfoques:

  * pueden ser sensibles a particularidades de la métrica
  * pueden ignorar relaciones complementarias entre algoritmos
  * pueden fallar al generalizar a datasets nuevos

### 🔍 La solución propuesta

El paper propone construir el ranking de algoritmos usando **reglas basadas en comparaciones par a par** de desempeño:

> En lugar de intentar predecir “el mejor algoritmo”, la idea es aprender **reglas meta** que predicen si A > B, B > C, etc., para cada par de algoritmos.

Esto genera un **ranking más estable** y con mejor calidad de ordenación cuando hay muchos algoritmos posibles.

---

## 🧠 ¿Qué significa *pairwise meta‑rules*?

* Una **meta‑regla par a par** es una regla que compara 2 algoritmos con base en meta‑features del dataset.

* Por ejemplo:

  * “Si el número de instancias es mayor a X y la entropía de la clase es menor a Y, entonces A es mejor que B”
  * Estas reglas se aprenden usando meta‑datos históricos.

* **Meta‑features del dataset** pueden ser:

  * número de atributos
  * número de instancias
  * proporción de clases
  * medidas estadísticas
  * etc.

---

## 🧪 ¿Cómo funciona el método?

1. **Recolectar meta‑datos históricos**

   * Para muchos datasets:
     `{dataset_meta_features, rendimiento_algoritmos}`

2. **Construir comparaciones parejas**

   * Para cada par `(Alg_i, Alg_j)`, crear ejemplos de entrenamiento:

     * Si Alg_i fue mejor → etiqueta “i > j”
     * Si Alg_j fue mejor → etiqueta “j > i”

3. **Entrenar meta‑clasificadores par a par**

   * Con meta‑features del dataset como entradas
   * Con la comparación (i mejor que j) como salida

4. **Construir ranking para un dataset nuevo**

   * Aplicar los meta‑clasificadores par a par
   * Combinar resultados para obtener el ranking final

---

## 🔧 ¿Qué datos / algoritmos se usan?

* El paper está típicamente evaluado con:

  * Datasets públicos (open benchmarks)
  * Algoritmos clásicos de clasificación
  * Las métricas de desempeño pueden ser accuracy u otras métricas relevantes

⚠️ El foco **no es un conjunto específico de datasets o pipelines**, sino **las reglas generadas**.

---

## 🧠 ¿Qué parte del Meta‑Learning aborda?

### ✔️ Lo que hace

* **Algorithm Ranking** usando meta‑learning
* **Construcción de reglas interpretables** para comparar pares de algoritmos
* Uso de **meta‑features del dataset** para alimentar esas reglas
* Evaluación de ranking en datasets de benchmark

### ❌ Lo que NO hace

* No genera pipelines completos
* No usa técnicas de AutoML avanzado
* No se focaliza en hiperparámetros
* Se centra en ranking de algoritmos individuales

---

## ▶️ ¿Qué aporta respecto a métodos clásicos?

Las contribuciones clave son:

### 📌 1. Mejor calidad de ranking

* Los rankings generados a partir de reglas par a par suelen ser más robustos y generalizables.
* Están menos afectados por ruido o problemas con métricas.

### 📌 2. Interpretabilidad

* Las reglas par a par pueden **interpretarse** fácilmente:

  * “Si la dimensionalidad es X y la clase está desequilibrada, entonces algoritmo A tiende a superar a B”.

### 📌 3. Escalabilidad

* Este enfoque puede escalar mejor a muchos algoritmos que tratar de predecir un ranking completo de una vez.

---

## 📈 Ejemplo simplificado

Si tenemos 3 algoritmos: A, B, C

| Dataset | A vs B | A vs C | B vs C |
| ------- | ------ | ------ | ------ |
| d1      | A > B  | A > C  | C > B  |
| d2      | B > A  | C > A  | C > B  |
| d3      | A > B  | C > A  | C > B  |

Entonces:

* Para cada par (A, B), (A, C), (B, C), entrenas un meta‑clasificador.
* Para un dataset nuevo:

  * Estimas si A > B, A > C, B > C
  * Combinas esas predicciones → ranking final

---

## 🧠 ¿Por qué es relevante para tu proyecto?

Este paper es muy útil si tu objetivo es:

* entrenar un sistema que **no solo elija un algoritmo**, sino que **genere un ranking** ordenado de algoritmos
* **comprender qué características de un dataset favorecen un algoritmo frente a otro**
* desarrollar meta‑learners **explicables**
* combinar resultados de muchos pares para producir una recomendación robusta

---

## 📝 Resumen para tu estado del arte

> *Sun & Pfahringer proponen un método de meta‑learning para ranking de algoritmos que se basa en reglas par a par entre algoritmos, construidas usando meta‑features de datasets. Este enfoque mejora la robustez e interpretabilidad del ranking frente a métodos convencionales de selección de algoritmos.*

---

## 📌 En una frase para tu proyecto

> *“Este trabajo usa comparaciones par a par entre algoritmos, basadas en meta‑features, para construir rankings más robustos y explicables para meta‑learning en selección de algoritmos.”*

---

# *Paper11: 🧠 Ranking Learning Algorithms: Using IBL and Meta-Learning on Accuracy and Time Results

**Link:** [🧠 anking Learning Algorithms: Using IBL and Meta-Learning on Accuracy and Time Results](state_of_the_art/Ranking_Learning_Algorithms_Using_IBL_and_Meta-Lea.pdf)


**Autores:** Pavel B. Brazdil, Carlos Soares, Joaquim Pinto da Costa
**Afiliación:** LIACC, University of Porto, Portugal
**Keywords:** algorithm recommendation, meta-learning, data characterization, ranking

---

## 📌 Contexto general

* El paper aborda **la selección de algoritmos de Machine Learning** mediante **meta-learning**, considerando **precisión y tiempo de ejecución**.
* El problema: elegir el algoritmo más adecuado para un dataset nuevo basándose en experiencias previas de otros datasets similares.

---

## 🎯 Idea central del paper

### 🔹 Problema al que responde

* No todos los datasets son iguales; un algoritmo puede funcionar bien en uno y mal en otro.
* Seleccionar manualmente un algoritmo puede ser lento o poco efectivo.
* El objetivo es generar un **ranking de algoritmos candidato** basado en características del dataset.

### 🔹 Solución propuesta

1. **Representación de datasets mediante meta-features**

   * Se elige un **pequeño conjunto de características de los datos** que influyen en el desempeño de los algoritmos, por ejemplo:

     * número de instancias
     * número de atributos
     * proporción de clases
     * otras medidas estadísticas o de complejidad del dataset

2. **Identificación de datasets similares**

   * Se usa **k-Nearest Neighbor (k-NN)** para encontrar datasets previos **similares al dataset actual** según las meta-features.

3. **Generación de ranking de algoritmos**

   * Se toman los **rendimientos de los algoritmos en los datasets similares**.
   * Se crea un **ranking multicriterio** que considera:

     * **Accuracy** (precisión)
     * **Tiempo de ejecución**

4. **Evaluación de rankings**

   * Se adapta una **metodología estadística para evaluar rankings**, ya que no es común trabajar con rankings directamente en ML.
   * Comparan su método con un **ranking base** y muestran mejoras significativas.

---

## 🧠 Qué aporta este paper

### ✔️ Principales aportes

1. **Uso de meta-learning para ranking multicriterio**

   * Integra **precisión y tiempo de ejecución** en un único ranking.
2. **Evaluación de rankings**

   * Proponen una metodología estadística general para evaluar rankings, aplicable a otros problemas de ranking.
3. **Recomendación interpretativa**

   * El método proporciona al usuario un **ranking de algoritmos candidato** en lugar de un único algoritmo.

### ✔️ Relevancia para meta-learning

* Muestra cómo se puede usar **información histórica de datasets** y **meta-features** para predecir el rendimiento de algoritmos en datasets nuevos.
* Aunque aquí se concentra en clasificación, el enfoque es **generalizable a combinaciones de métodos o estrategias más complejas**.

---

## 🔹 Ejemplo conceptual

Supongamos que tienes 3 algoritmos: A, B y C y un dataset nuevo D:

1. Encuentras los datasets similares a D según meta-features.
2. Observas cómo A, B y C funcionaron en esos datasets.
3. Generas un ranking basado en desempeño y tiempo:

   * Dataset similar 1: B > A > C
   * Dataset similar 2: A > C > B
4. Combinando resultados → ranking final recomendado para D.

---

## 🧠 Por qué es útil para tu proyecto

* **Selección de algoritmos basada en meta-features** → directamente aplicable a sistemas de recomendación de pipelines.
* **Ranking multicriterio** → útil si quieres considerar **más de una métrica** (por ejemplo, precisión vs tiempo de entrenamiento).
* **Framework general de meta-learning** → se puede extender a:

  * selección de pipelines completos
  * selección de hiperparámetros
  * integración en AutoML

---

En resumen:

> Este trabajo propone un enfoque de meta-learning basado en k-NN para recomendar un **ranking de algoritmos de clasificación**, usando meta-features de datasets y evaluando precisión y tiempo. La metodología permite generar rankings más precisos que métodos base y es aplicable a problemas de recomendación más amplios.

---

# Paper12 : 🧠 Tunability: Importance of Hyperparameters of Machine Learning Algorithms

**Link:** [🧠 Tunability: Importance of Hyperparameters of Machine Learning Algorithms](state_of_the_art/Tunability_%20Importance%20of%20Hyperparameters%20of%20Machine.pdf)

**Autores:** Philipp Probst, Anne-Laure Boulesteix, Bernd Bischl
**Keywords:** hyperparameter tuning, tunability, meta-learning, OpenML, benchmarking

---

## 📌 Contexto general

* Muchos algoritmos de **Machine Learning supervisado** dependen de **hiperparámetros** que deben configurarse antes de entrenar.

* Elegir valores adecuados puede mejorar significativamente el rendimiento.

* Opciones comunes:

  1. Valores por defecto del software
  2. Configuración manual por el usuario
  3. Optimización automática mediante tuning

* El paper se centra en **cuantificar la importancia de los hiperparámetros** y evaluar qué tan “tunables” son.

---

## 🎯 Idea central del paper

### 🔹 Problema al que responde

* No todos los hiperparámetros son igual de importantes para todas las tareas.
* Necesidad de saber:

  * Qué hiperparámetros realmente afectan el rendimiento
  * Cuándo vale la pena realizar tuning costoso
* Esto es crucial para **meta-learning y AutoML**, porque ayuda a priorizar recursos y decidir si ajustar parámetros mejora el rendimiento.

### 🔹 Solución propuesta

1. **Formalización del concepto de tunabilidad**

   * Definen estadísticamente cuánto impacta un hiperparámetro en el rendimiento esperado del algoritmo.
   * Introducen medidas generales para cuantificar la tunabilidad de cada parámetro.

2. **Benchmarking a gran escala**

   * Usan **38 datasets de OpenML**.
   * Evaluación de **6 algoritmos comunes**.
   * Para cada algoritmo, generan muchas configuraciones de hiperparámetros y comparan rendimientos.

3. **Análisis y recomendaciones**

   * Identifican **valores por defecto basados en datos**.
   * Determinan cuáles parámetros realmente influyen en el rendimiento.
   * Permiten decidir cuándo **vale la pena realizar tuning** y cuáles se pueden dejar en default.

---

## 🧠 Qué aporta este paper

### ✔️ Principales aportes

1. **Medidas cuantitativas de tunabilidad**

   * Permiten saber la importancia relativa de cada hiperparámetro.
   * Facilitan comparaciones entre algoritmos.

2. **Benchmarking extensivo**

   * Datos públicos de OpenML
   * Amplio análisis de hiperparámetros en varios datasets

3. **Orientación práctica**

   * Para AutoML y sistemas de meta-learning:

     * Identificar qué hiperparámetros priorizar
     * Reducir el tiempo computacional evitando tuning innecesario

---

## 🔹 Ejemplo conceptual

* Algoritmo: Random Forest
* Hiperparámetros:

  * n_estimators, max_depth, min_samples_split
* Resultado:

  * n_estimators → bajo impacto en performance (poco tunable)
  * max_depth → alto impacto (muy tunable)
  * min_samples_split → impacto moderado
* Con esto, un sistema de AutoML puede **centrarse en tunear max_depth y min_samples_split** y dejar n_estimators por defecto, ahorrando tiempo.

---

## 🧠 Relevancia para meta-learning y AutoML

* Permite **priorizar hiperparámetros críticos** en un proceso de recomendación de algoritmos o pipelines.
* Se puede integrar en sistemas como:

  * **AlphaD3M** → priorizar hiperparámetros durante síntesis de pipelines
  * **PIPES** → construir meta-datasets con foco en parámetros más importantes
* Mejora la **eficiencia de tuning automático** y ayuda a decidir si vale la pena un ajuste exhaustivo.

---

## 🔹 Resumen para tu estado del arte

> *Probst et al. formalizan el concepto de “tunabilidad” de los hiperparámetros y proporcionan medidas para evaluar su impacto en el rendimiento de algoritmos. Usando datasets de OpenML, identifican qué parámetros son críticos y cuáles pueden dejarse en valores por defecto, lo que es útil para meta-learning y AutoML.*

---

