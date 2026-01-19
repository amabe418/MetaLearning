import pandas as pd

def rank_algorithms_by_task(task_id: str):
    """
    Dado un dataset/task_id, lee varios CSV de algoritmos y devuelve
    un ranking de algoritmos basado en su mejor Fold Accuracy para ese task.
    """
    pathFolder = "../data/top_raw_target_representation/"

    csv_paths = {
        "AdaBoost": "adaboost_target_representation.csv",
        "RandomForest": "random_forest_target_representation.csv",
        "SVM": "libsvm_svc_target_representation.csv"
    }

    # Diccionario para guardar el mejor accuracy de cada algoritmo
    best_scores = {}

    for alg_name, csv_file in csv_paths.items():
        full_path = f"{pathFolder}{csv_file}"
        df = pd.read_csv(full_path)

        # Filtrar por task_id
        df_task = df[df['task_id'] == task_id]

        if df_task.empty:
            print(f"⚠️ No se encontraron resultados para {alg_name} en {task_id}")
            continue

        # Tomar el máximo Accuracy
        max_acc = df_task['predictive_accuracy'].max()
        best_scores[alg_name] = max_acc

    # Convertir a DataFrame para ordenar
    ranking_df = pd.DataFrame(
        list(best_scores.items()),
        columns=["Algorithm", "predictive_accuracy"]
    ).sort_values(by="predictive_accuracy", ascending=False)

    return ranking_df


if __name__ == "__main__":
    task_id = 12
    ranking = rank_algorithms_by_task(task_id)
    print(f"Ranking de algoritmos para {task_id}:")
    print(ranking)
