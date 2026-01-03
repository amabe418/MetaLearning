import pandas as pd
import json
import openml
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# le pasas un conjunto de task y te extrae los runs y los almacena en un csv

def extract_runs(task_ids, path_csv = "meta-dataset.csv", n_top =10, metric = "f_measure"):

    rows = []

    for task_id in task_ids:

        task = openml.tasks.get_task(task_id)

        evaluations = openml.evaluations.list_evaluations(
            metric, 
            tasks=[task_id], 
            output_format="dataframe", 
            size=None
        )

        # Ordenamos por metric descendente
        evaluations_sorted = evaluations.sort_values(by="value", ascending=False)

        # Eliminamos repeticiones por flow_id, manteniendo la mejor evaluación
        evaluations_unique = evaluations_sorted.drop_duplicates(subset=["flow_id"], keep="first")

        # Tomamos solo los n_top
        top_evaluations = evaluations_unique.head(n_top)

        print(f"Task {task_id}: {len(top_evaluations)} evaluaciones únicas seleccionadas")

        for _, row_eval in top_evaluations.iterrows():
            run = openml.runs.get_run(row_eval.run_id)
            setup = openml.setups.get_setup(run.setup_id)
            flow = openml.flows.get_flow(setup.flow_id)

            row = {
                "dataset_id": task.dataset_id,
                "task_id": task.task_id,
                "flow_id": flow.flow_id,
                "f_measure": row_eval.value,
                "algorithm": flow.name,
                "hyperparameters": setup.parameters
            }
            rows.append(row)
            print(row)

    meta_dataset = pd.DataFrame(rows)

    meta_dataset['hyperparameters'] = meta_dataset['hyperparameters'].apply(_param_to_dict)
    meta_dataset['hyperparameters_json'] = meta_dataset['hyperparameters'].apply(json.dumps)

    meta_dataset.to_csv(path_csv, index=False)
    print("CSV guardado correctamente")

# Convertimos hyperparameters a dict y luego a JSON
def _param_to_dict(param_obj):
    if param_obj is None:
        return {}
    return {k: v.value for k, v in param_obj.items()}