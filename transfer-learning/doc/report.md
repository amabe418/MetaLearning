# Reporte: Preparación de Datos para FSBO

## 1. Introducción a FSBO

FSBO (Few-Shot Bayesian Optimization) es una técnica de meta-learning que aplica optimización bayesiana con transferencia de conocimiento entre tareas, permitiendo encontrar buenas configuraciones de hiperparámetros con muy pocas evaluaciones en nuevas tareas. El método fue propuesto por Wistuba & Grabocka (ICLR 2021).

El dataset de entrenamiento en FSBO consiste en un meta-dataset de tareas de optimización, donde cada tarea está representada por pares `(X, y)`:
- **X**: Configuraciones de hiperparámetros (normalizadas)
- **y**: Métricas de rendimiento correspondientes (accuracy, AUC, etc.)

---

## 2. Datos Disponibles Inicialmente

Se disponía de datos preprocesados en `transfer-learning/data/` con la siguiente estructura:

### 2.1 Espacios de Configuración (ConfigSpace)
Archivos JSON que definen los espacios de hiperparámetros para cada algoritmo:

| Archivo | Algoritmo | Hiperparámetros |
|---------|-----------|-----------------|
| `adaboost_configspace.json` | AdaBoost | 5 |
| `random_forest_configspace.json` | Random Forest | 6 |
| `libsvm_svc_configspace.json` | SVM (LibSVM) | 9 |
| `autosklearn_configspace.json` | Auto-sklearn | ~130 |

### 2.2 Representaciones de Hiperparámetros
Archivos CSV con configuraciones preprocesadas (normalizadas y con one-hot encoding):

| Archivo | Muestras | Tareas | Features |
|---------|----------|--------|----------|
| `adaboost_target_representation.csv` | 4,665 | 64 | 8 |
| `random_forest_target_representation.csv` | 10,746 | 64 | 10 |
| `libsvm_svc_target_representation.csv` | 4,523 | 64 | 12 |
| `autosklearn_target_representation.csv` | 6,481 | 64 | 222 |

### 2.3 Problema Identificado

**Los archivos de representación NO contenían métricas de rendimiento (y).** Solo incluían:
- `task_id`: Identificador de la tarea/dataset
- Hiperparámetros normalizados

Sin los valores de rendimiento, no era posible entrenar, validar ni testear un modelo FSBO.

---

## 3. Solución Implementada: Generación de Métricas Sintéticas

Para validar la implementación de FSBO en este proyecto académico, se generaron métricas de rendimiento sintéticas mediante el script `scripts/generate_synthetic_scores.py`.

### 3.1 Estrategia de Generación

Las métricas sintéticas se generaron con las siguientes características para simular comportamiento realista:

1. **Superficie de respuesta**: Se creó una función que relaciona los hiperparámetros con el rendimiento, combinando:
   - Componente lineal: `X @ weights`
   - Componente de interacción: `X[:,0] * X[:,1]`

2. **Diferentes óptimos por tarea**: Cada `task_id` tiene pesos únicos generados mediante hash del ID, simulando que diferentes datasets tienen diferentes configuraciones óptimas.

3. **Rango realista**: Los scores se generan en el rango [0.50, 0.99], típico de métricas como accuracy.

4. **Ruido gaussiano**: Se añade ruido con σ=0.03 para simular la variabilidad inherente a evaluaciones reales.

### 3.2 Justificación Académica

La generación de datos sintéticos es una práctica común en investigación para:
- Validar implementaciones antes de usar datos reales
- Realizar pruebas de concepto
- Depurar y ajustar código

Para investigación publicable, se recomienda utilizar datos reales de benchmarks como OpenML o HPO-B.

---

## 4. Datos Resultantes

Los nuevos archivos se guardaron en `transfer-learning/data/representation_with_scores/`:

| Archivo | Muestras | Tareas | Score medio | Score std |
|---------|----------|--------|-------------|-----------|
| `adaboost_target_representation_with_scores.csv` | 4,665 | 64 | 0.744 | 0.085 |
| `random_forest_target_representation_with_scores.csv` | 10,746 | 64 | 0.747 | 0.083 |
| `libsvm_svc_target_representation_with_scores.csv` | 4,523 | 64 | 0.743 | 0.079 |
| `autosklearn_target_representation_with_scores.csv` | 6,481 | 64 | 0.744 | 0.076 |

### 4.1 Estructura de los Archivos

```csv
task_id,hiperparámetro_1,hiperparámetro_2,...,accuracy
3,1.237,-0.007,...,0.8408
3,1.585,-1.605,...,0.8528
...
```

### 4.2 Verificación de Requisitos para FSBO

| Requisito | Estado | Detalle |
|-----------|--------|---------|
| Mínimo 5 evaluaciones por tarea | ✅ | Mín: 11-26 por tarea |
| Múltiples tareas para división | ✅ | 64 tareas únicas |
| División train/val/test | ✅ | Posible: ~45/10/9 |
| Hiperparámetros normalizados | ✅ | Ya preprocesados |
| Métrica de rendimiento | ✅ | Columna `accuracy` añadida |

---

## 5. Resumen de Totales

| Métrica | Valor |
|---------|-------|
| **Muestras totales** | 26,415 |
| **Tareas únicas** | 64 |
| **Algoritmos cubiertos** | 4 |
| **Promedio muestras/tarea** | ~103 |

---

## 6. Próximos Pasos

Con los datos completos, ahora es posible:

1. **Entrenar** el modelo Deep Kernel GP en tareas fuente
2. **Validar** la convergencia y ajustar hiperparámetros del entrenamiento
3. **Testear** con leave-one-task-out para evaluar transferencia
4. **Comparar** con baselines (Random Search, GP vanilla, etc.)

---

## 7. Referencias

- Wistuba, M., & Grabocka, J. (2021). *Few-Shot Bayesian Optimization with Deep Kernel Surrogates*. ICLR 2021.
- HPO-B: A Large-Scale Benchmark for Hyperparameter Optimization
- OpenML: An Open Platform for Machine Learning
