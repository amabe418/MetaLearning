import pandas as pd

import openml

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


dataset_ids = [
    3, 6, 12, 14, 16, 18, 21, 22, 23, 24, 26, 28, 30, 31, 32, 36, 38, 44, 46,
    57, 60, 179, 180, 181, 182, 184, 185, 273, 293, 300, 351, 354, 357, 389,
    390, 391, 392, 393, 395, 396, 398, 399, 401, 554, 679, 715, 718, 720, 722,
    723, 727, 728, 734, 735, 737, 740, 741, 743, 751, 752, 761, 772, 797, 799,
    803, 806, 807, 813, 816, 819, 821, 822, 823, 833, 837, 843, 845, 846, 847,
    849, 866, 871, 881, 897, 901, 903, 904, 910, 912, 913, 914, 917, 923, 930,
    934, 953, 958, 959, 962, 966, 971, 976, 977, 978, 979, 980, 991, 993, 995,
    1000, 1002, 1018, 1019, 1020, 1021, 1036, 1040, 1041, 1049, 1050, 1053,
    1056, 1067, 1068, 1069, 1111, 1112, 1114, 1116, 1119, 1120, 1128, 1130,
    1134, 1138, 1139, 1142, 1146, 1161, 1166,
]


tasks = openml.tasks.list_tasks(
    task_type=openml.tasks.TaskType.SUPERVISED_CLASSIFICATION,
    status="all",
    output_format="dataframe",
)

# Query only those with holdout as the resampling startegy.
tasks = tasks.query('estimation_procedure == "33% Holdout set"')

task_ids = []
for did in dataset_ids:
    tasks_ = list(tasks.query("did == {}".format(did)).tid)
    if len(tasks_) >= 1:  # if there are multiple task, take the one with lowest ID (oldest).
        task_id = min(tasks_)
    else:
        raise ValueError(did)

    # Optional - Check that the task has the same target attribute as the
    # dataset default target attribute
    # (disabled for this example as it needs to run fast to be rendered online)
    # task = openml.tasks.get_task(task_id)
    # dataset = task.get_dataset()
    # if task.target_name != dataset.default_target_attribute:
    #     raise ValueError(
    #         (task.target_name, dataset.default_target_attribute)
    #     )

    task_ids.append(task_id)

assert len(task_ids) == 140
task_ids.sort()

# These are the tasks to work with:
print(task_ids)

################# PARTE IMPORTANTE ###############

task_id = task_ids[0]
task = openml.tasks.get_task(task_id)

evaluations = openml.evaluations.list_evaluations(
    "f_measure",
    tasks=[task_id],
    output_format="dataframe",
    size = None
)

pd.set_option("display.max_columns", None)  # muestra todas las columnas
pd.set_option("display.max_colwidth", None)  # muestra contenido completo de cada celda

# for fid in evaluations["flow_id"].unique():
#     flow = openml.flows.get_flow(fid)
#     print(fid, flow.name, flow.model)

for run_id in evaluations['run_id'].head(5):  # primeros 5 runs
    run = openml.runs.get_run(run_id)
    setup = openml.setups.get_setup(run.setup_id)
    flow = openml.flows.get_flow(setup.flow_id)  # obtenemos el flow
    print(run_id, flow.name, setup.parameters)


print(evaluations.head())
