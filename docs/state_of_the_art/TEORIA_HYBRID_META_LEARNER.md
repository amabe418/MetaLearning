# Teoría y Diseño del Hybrid Meta-Learner

## 📚 Resumen Ejecutivo

Este documento explica la teoría, diseño e implementación del **Hybrid Meta-Learner**, un algoritmo de meta-learning que combina las mejores técnicas del estado del arte según el survey de Vanschoren (2019). El algoritmo está diseñado para recomendar algoritmos de machine learning y sus configuraciones para nuevos datasets basándose en experiencia previa.

---

## 🎯 Objetivo del Algoritmo

El Hybrid Meta-Learner tiene como objetivo:
1. **Recomendar algoritmos** apropiados para un nuevo dataset
2. **Predecir el rendimiento** esperado de cada algoritmo
3. **Acelerar la búsqueda** usando conocimiento de tareas similares
4. **Proporcionar explicaciones** de las recomendaciones

---

## 🏗️ Fundamentos Teóricos

### 1. Meta-Learning: Definición y Principios

**Meta-learning** (aprendizaje de aprendizaje) es el proceso de aprender de la experiencia previa con múltiples tareas de aprendizaje para mejorar el rendimiento en nuevas tareas.

**Principios clave:**
- **Transferencia de conocimiento:** Aprovechar lo aprendido en tareas previas
- **Similitud de tareas:** Tareas similares requieren algoritmos similares
- **Meta-datos:** Información sobre tareas, algoritmos y su rendimiento
- **Generalización:** Aprender patrones generalizables entre tareas

### 2. Tipos de Meta-Datos

Según el survey, existen tres tipos principales de meta-datos:

1. **Evaluaciones de Modelos (P):** Rendimiento de configuraciones en tareas previas
2. **Propiedades de Tareas (M):** Meta-features que caracterizan los datasets
3. **Modelos Previos (L):** Parámetros y estructuras de modelos entrenados

Nuestro algoritmo utiliza principalmente los tipos 1 y 2, siendo más aplicable a datos tabulares.

---

## 🔬 Componentes del Algoritmo

### Componente 1: Búsqueda de Tareas Similares

**Base Teórica:** Sección 3.3 del survey - "Warm-Starting Optimization from Similar Tasks"

**Implementación:**
- Usa **k-Nearest Neighbors (k-NN)** en el espacio de meta-features
- Distancia euclidiana en espacio normalizado
- Encuentra las k tareas más similares a la nueva tarea

**Justificación:**
- Tareas con meta-features similares tienden a requerir algoritmos similares
- Permite transferir conocimiento de manera eficiente
- Base para warm-starting

**Referencias del documento:**
- Gomes et al. (2012): Usan L1 distance entre meta-features
- Feurer et al. (2014): Warm-starting con tareas similares

### Componente 2: Warm-Starting con Configuraciones de Tareas Similares

**Base Teórica:** Sección 3.3 - "Warm-Starting Optimization from Similar Tasks"

**Implementación:**
- Pre-computa las mejores configuraciones de cada tarea
- Para una nueva tarea, obtiene configuraciones de tareas similares
- Ponderación por similitud

**Justificación:**
- Acelera la convergencia hacia buenas soluciones
- Reduce el espacio de búsqueda
- Aprovecha conocimiento previo de manera explícita

**Referencias:**
- Feurer et al. (2014, 2015): Warm-starting en autosklearn
- Gomes et al. (2012): Inicialización de algoritmos genéticos

### Componente 3: Meta-Models para Predicción de Rendimiento

**Base Teórica:** Sección 3.4.2 - "Performance Prediction"

**Implementación:**
- Un **Random Forest Regressor** por algoritmo
- Entrenado en: meta-features → rendimiento del algoritmo
- Predice rendimiento esperado para nueva tarea

**Justificación:**
- Random Forest es robusto y efectivo según el survey
- Permite estimar rendimiento sin evaluar
- Útil para ranking y selección

**Referencias:**
- Reif et al. (2014): Meta-regressors para predicción de accuracy
- Guerra et al. (2008): SVM meta-regressors

### Componente 4: Meta-Models para Ranking

**Base Teórica:** Sección 3.4.1 - "Ranking"

**Implementación:**
- Un modelo por algoritmo que predice su posición en ranking
- Alternativamente, ranking basado en predicciones de rendimiento
- Combina múltiples señales

**Justificación:**
- Ranking es más robusto que valores absolutos de rendimiento
- Mejor para comparar algoritmos
- Útil cuando las escalas de rendimiento varían entre tareas

**Referencias:**
- Sun & Pfahringer (2013): ART Forests para ranking
- Brazdil et al. (2003): Rankings basados en rendimiento

### Componente 5: Active Testing Iterativo

**Base Teórica:** Sección 2.3.1 - "Relative Landmarks" y "Active Testing"

**Implementación:**
- Selecciona el siguiente algoritmo a evaluar basándose en:
  - Predicción de rendimiento
  - Similitud con tareas previas
  - Probabilidad de superar al mejor actual

**Justificación:**
- Eficiente en uso de recursos computacionales
- Enfoque iterativo que aprende mientras evalúa
- Combina múltiples fuentes de información

**Referencias:**
- Leite et al. (2012): Active Testing
- Fürnkranz & Petrak (2001): Relative Landmarks

---

## 🎨 Decisiones de Diseño

### Decisión 1: Uso de RobustScaler en lugar de StandardScaler

**Razón:**
- Los meta-features pueden tener outliers
- RobustScaler usa mediana y rango intercuartílico (más robusto)
- Mejor generalización en presencia de valores extremos

### Decisión 2: Combinación de Múltiples Señales

**Fórmula de Score Combinado:**
```
combined_score = 0.6 * normalized_performance + 
                 0.3 * normalized_rank + 
                 0.1 * warm_start_boost
```

**Razón:**
- **60% rendimiento:** La señal más directa e importante
- **30% ranking:** Proporciona contexto relativo
- **10% warm-start:** Boost adicional de tareas similares

**Alternativas consideradas:**
- Pesos iguales: No captura la importancia relativa
- Solo rendimiento: Ignora información valiosa de similitud
- Solo ranking: Menos preciso para valores absolutos

### Decisión 3: Número de Tareas Similares (n_similar_tasks = 5)

**Razón:**
- Balance entre información y ruido
- Múltiples tareas similares aumentan robustez
- Demasiadas tareas diluyen la señal de similitud

**Evidencia del documento:**
- Feurer et al. (2014): Usan top-d tareas similares (d pequeño)
- Gomes et al. (2012): k-NN con k pequeño

### Decisión 4: Random Forest como Meta-Model Base

**Razón:**
- Mencionado como efectivo en múltiples estudios del survey
- Maneja bien relaciones no lineales
- Robusto a outliers y características irrelevantes
- No requiere tuning extensivo

**Alternativas consideradas:**
- **SVM:** Menos escalable, requiere más tuning
- **Neural Networks:** Overkill para este problema, requiere más datos
- **Linear Models:** Demasiado simples para relaciones complejas

### Decisión 5: Enfoque Híbrido vs. Enfoques Puros

**Por qué híbrido:**
- **Solo warm-starting:** Ignora predicciones de rendimiento
- **Solo predicción:** Ignora conocimiento de tareas similares
- **Híbrido:** Combina lo mejor de ambos mundos

**Evidencia:**
- El documento muestra que combinar técnicas es efectivo
- Feurer et al. (2015): autosklearn combina múltiples técnicas
- Wistuba et al. (2018): Combinan surrogate models y warm-starting

### Decisión 6: Normalización de Rendimientos

**Implementación:**
- Asume rango 0-1 para normalización
- En práctica, puede ajustarse según el dominio

**Razón:**
- Permite combinar señales en diferentes escalas
- Ranking ya está normalizado (1/rank)
- Warm-start boost normalizado por similitud

---

## 📊 Flujo del Algoritmo

### Fase 1: Entrenamiento

```
1. Recibir meta-features y rendimientos de tareas previas
2. Normalizar meta-features
3. Entrenar modelo de similitud (k-NN)
4. Entrenar predictores de rendimiento (uno por algoritmo)
5. Entrenar modelos de ranking (opcional)
6. Pre-computar configuraciones de warm-starting
```

### Fase 2: Recomendación para Nueva Tarea

```
1. Extraer meta-features de nueva tarea
2. Normalizar meta-features
3. Encontrar tareas similares (k-NN)
4. Obtener recomendaciones de warm-starting
5. Predecir rendimiento de todos los algoritmos
6. Predecir ranking de todos los algoritmos
7. Combinar señales en score final
8. Retornar top-k recomendaciones
```

### Fase 3: Active Testing (Opcional)

```
1. Evaluar algoritmo recomendado
2. Actualizar mejor algoritmo actual
3. Seleccionar siguiente algoritmo usando:
   - Predicción de rendimiento
   - Similitud con tareas donde candidato supera al mejor actual
4. Repetir hasta presupuesto agotado o convergencia
```

---

## 🔍 Ventajas del Diseño

### 1. **Robustez**
- Combina múltiples fuentes de información
- No depende de una sola técnica
- Maneja bien casos edge

### 2. **Eficiencia**
- Warm-starting acelera búsqueda
- Active testing reduce evaluaciones innecesarias
- Pre-computación de configuraciones

### 3. **Interpretabilidad**
- Proporciona explicaciones (tareas similares, razones)
- Transparente en sus decisiones
- Permite debugging y análisis

### 4. **Flexibilidad**
- Puede desactivar componentes (warm-start, ranking)
- Adaptable a diferentes dominios
- Extensible con nuevas técnicas

### 5. **Basado en Evidencia**
- Todas las técnicas están respaldadas por el survey
- Combinaciones probadas en la literatura
- Parámetros justificados

---

## ⚠️ Limitaciones y Consideraciones

### Limitación 1: Requiere Meta-Datos Previos

**Problema:** Necesita evaluaciones previas de algoritmos en múltiples tareas.

**Solución:**
- Usar repositorios como OpenML
- Evaluar algoritmos en conjunto de tareas base
- Cold-start problem para primera tarea (usar rankings globales)

### Limitación 2: Asume Similitud de Tareas

**Problema:** Si nueva tarea es muy diferente, transferencia puede fallar.

**Solución:**
- Incluir predicción de rendimiento (no solo similitud)
- Detectar cuando similitud es baja y confiar más en predicción
- Fallback a rankings globales

### Limitación 3: Escala de Rendimiento

**Problema:** Asume normalización 0-1, puede no ser siempre válida.

**Solución:**
- Calibrar según dominio (accuracy, F1, AUC, etc.)
- Usar ranking como señal adicional (más robusto)
- Ajustar pesos según confianza en normalización

### Limitación 4: Complejidad Computacional

**Problema:** Entrenar múltiples modelos puede ser costoso.

**Solución:**
- Pre-entrenar modelos una vez
- Reutilizar para múltiples nuevas tareas
- Optimizar hiperparámetros de meta-models offline

---

## 🚀 Mejoras Futuras

### 1. **Meta-Features de Landmarking**
- Evaluar algoritmos simples (1NN, Tree, Linear, NB) en cada dataset
- Usar su rendimiento como meta-features adicionales
- Mejora la caracterización de tareas

### 2. **Surrogate Models con Gaussian Processes**
- Modelos más sofisticados para predicción de rendimiento
- Capturan incertidumbre (útil para active testing)
- Mejor para optimización bayesiana

### 3. **Ensemble de Meta-Models**
- Combinar múltiples meta-models (voting, stacking)
- Aumenta robustez y precisión
- Similar a ART Forests del documento

### 4. **Aprendizaje de Pesos**
- Aprender pesos de combinación en lugar de fijos
- Adaptar según características de la tarea
- Meta-learning de segundo nivel

### 5. **Transfer de Hiperparámetros**
- No solo recomendar algoritmo, sino también hiperparámetros
- Usar configuraciones completas de tareas similares
- Integrar con optimización bayesiana

---

## 📚 Referencias Clave del Documento

### Sobre Warm-Starting:
- **Feurer et al. (2014, 2015):** Warm-starting en autosklearn
- **Gomes et al. (2012):** Inicialización de algoritmos genéticos
- **Reif et al. (2012):** Warm-starting con meta-features

### Sobre Meta-Models:
- **Brazdil et al. (2009):** Libro clásico sobre meta-learning
- **Sun & Pfahringer (2013):** ART Forests para ranking
- **Reif et al. (2014):** Meta-regressors para predicción

### Sobre Active Testing:
- **Leite et al. (2012):** Active Testing con relative landmarks
- **Fürnkranz & Petrak (2001):** Relative landmarks

### Sobre Combinación de Técnicas:
- **Wistuba et al. (2018):** Combinación de surrogate models y warm-starting
- **Feurer et al. (2015):** autosklearn como sistema híbrido

---

## 🎓 Conclusiones

El **Hybrid Meta-Learner** representa una implementación práctica y bien fundamentada de las mejores técnicas de meta-learning según el estado del arte. Combina:

1. ✅ **Búsqueda de similitud** para transferencia de conocimiento
2. ✅ **Warm-starting** para acelerar búsqueda
3. ✅ **Predicción de rendimiento** para estimación precisa
4. ✅ **Ranking** para comparación robusta
5. ✅ **Active testing** para eficiencia

El diseño es:
- **Teóricamente sólido:** Basado en survey académico
- **Prácticamente útil:** Aplicable a datos tabulares reales
- **Extensible:** Permite agregar nuevas técnicas
- **Interpretable:** Proporciona explicaciones

Este algoritmo proporciona una base sólida para sistemas de AutoML y selección automática de algoritmos, especialmente cuando se tiene acceso a meta-datos de experimentos previos (como OpenML).

---

## 📝 Notas de Implementación

### Dependencias:
- `scikit-learn`: Para modelos de ML y preprocesamiento
- `numpy` y `pandas`: Para manipulación de datos
- Compatible con estructura existente del proyecto

### Uso Típico:
```python
from src.learner import HybridMetaLearner

# Inicializar
learner = HybridMetaLearner(
    algorithms=['RandomForest', 'SVM', 'KNN'],
    n_similar_tasks=5,
    use_warm_start=True,
    use_ranking=True
)

# Entrenar
learner.train(meta_features_df, performance_df, task_ids)

# Recomendar para nueva tarea
recommendations = learner.recommend_algorithms(new_meta_features, top_k=5)

# Active testing
next_algorithm = learner.active_testing_step(
    new_meta_features,
    evaluated_algorithms=['RandomForest'],
    evaluated_performances={'RandomForest': 0.85}
)
```

---

**Autor:** Basado en el análisis del survey "Meta-Learning: A Survey" de Joaquin Vanschoren  
**Fecha:** 2024  
**Versión:** 1.0

