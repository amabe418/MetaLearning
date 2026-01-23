"""
baseline.py

Este script implementa un **baseline global de ranking de algoritmos** basado en la
precisión de cada algoritmo en múltiples datasets, usando el enfoque de **Adjusted Ratio
of Ratios (ARR)** simplificado, sin considerar tiempos de ejecución. 

Objetivo:
----------
Construir un ranking global de algoritmos que refleje su desempeño relativo de forma
robusta, evitando que un buen resultado aislado en un dataset influyas excesivamente
en la posición global de un algoritmo.

Metodología:
-------------
1. **Selección del mejor desempeño por dataset**:
   - Para cada algoritmo, se toma la mejor precisión registrada en cada dataset. 
   - Esto asegura que solo se considere el desempeño óptimo de cada configuración.

2. **Cálculo de ratios por pares (SRR, Success Rate Ratios)**:
   - Para cada dataset d_i y cada par de algoritmos (a_p, a_q), se calcula:
     
       SRR_{d_i}(a_p, a_q) = Accuracy(a_p, d_i) / Accuracy(a_q, d_i)
     
   - Esto mide la ventaja relativa de un algoritmo sobre otro en ese dataset.

3. **Agregación a través de los datasets (media geométrica)**:
   - Para cada par de algoritmos, se combinan los ratios de todos los datasets usando
     la media geométrica:
     
       ARR(a_p, a_q) = ( Π_{i=1}^n SRR_{d_i}(a_p, a_q) )^(1/n)
     
   - La media geométrica garantiza consistencia: ARR(a_p, a_q) = 1 / ARR(a_q, a_p).

4. **Cálculo del puntaje final por algoritmo**:
   - Para cada algoritmo a_p, se promedian sus ARR frente a todos los demás:
     
       Puntaje_Final(a_p) = ( Σ_{q=1}^m ARR(a_p, a_q) ) / m
     
   - El ranking global se obtiene ordenando los algoritmos según este puntaje.

Ventaja:
---------
Este enfoque compara los algoritmos de forma relativa y robusta, premiando a aquellos
que mantienen un desempeño consistente y evitando que resultados excepcionales en un
único dataset dominen el ranking global. Se considera un **baseline sólido** para
comparaciones estadísticas posteriores, por ejemplo con tests de Friedman y Dunn.
"""


import pandas as pd
from functools import reduce
import numpy as np

def load_best_accuracy(pathFolder:str,csv_paths: dict):
    """
    csv_paths: dict con key=algoritmo, value=ruta CSV
    Retorna un DataFrame con columnas: task_id + cada algoritmo
    """
    best_acc = []
    
    for algo, path in csv_paths.items():
        df = pd.read_csv(pathFolder+path)
        tmp = (
            df.groupby("task_id")["predictive_accuracy"]
            .max()
            .reset_index()
            .rename(columns={"predictive_accuracy": algo})
        )
        best_acc.append(tmp)
    
    # Merge por task_id
    acc_table = reduce(lambda left, right: pd.merge(left, right, on="task_id"), best_acc)
    return acc_table





def compute_srr(acc_table: pd.DataFrame):
    """
    Devuelve un diccionario con la puntuación final de cada algoritmo
    usando la media geométrica de los ratios por pares (SRR)
    """
    algorithms = [col for col in acc_table.columns if col != "task_id"]
    datasets = acc_table["task_id"].values

    SRR = {
        d: {
            a_p: {
                a_q: acc_table.loc[acc_table["task_id"] == d, a_p].values[0]
                / acc_table.loc[acc_table["task_id"] == d, a_q].values[0]
                for a_q in algorithms
            }
            for a_p in algorithms
        }
        for d in datasets
    }

    ARR = {}

    for a_p in algorithms:
        ARR[a_p] = {}
        for a_q in algorithms:
            ratios = [SRR[d][a_p][a_q] for d in datasets]
            ARR[a_p][a_q] = np.exp(np.mean(np.log(ratios)))

    final_scores = {}

    m = len(algorithms)
    for a_p in algorithms:
        final_scores[a_p] = sum(ARR[a_p][a_q] for a_q in algorithms) / m


    return final_scores




def main():

    pathFolder = "../data/top_raw_target_representation/"

    csv_paths = {
        "AdaBoost": "adaboost_target_representation.csv",
        "RandomForest": "random_forest_target_representation.csv",
        "SVM": "libsvm_svc_target_representation.csv"
    }
    
    acc_table = load_best_accuracy(pathFolder,csv_paths)
    
    global_min_acc = acc_table.drop(columns=["task_id"]).min().min()
    acc_table = acc_table.fillna(global_min_acc)
    acc_table = acc_table.replace(0, 1e-10) 

    scores = compute_srr(acc_table)
    
    ranking = (
        pd.DataFrame.from_dict(scores, orient="index", columns=["score"])
        .sort_values("score", ascending=False)
    )

    print("=== GLOBAL ARR RANKING ===")
    print(ranking)



if __name__ == "__main__":
    main()
