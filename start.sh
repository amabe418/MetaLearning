python main.py task=task1 openml_tid=3 task.ndcg=15 metafeature=metafeatx data_path=${PWD}/data pipeline=adaboost


python main.py task=task2 task.nb_iterations=5 openml_tid=3 metafeature=metafeatx pipeline=random_forest data_path=${PWD}/data/