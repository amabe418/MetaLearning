from config import benchmarks
import openml
import pandas as pd
from notebooks.tasks import extract_runs

def export_benchmark_to_csv(benchmark_name, csv_path="meta_dataset.csv"):
    bench_cfg = benchmarks[benchmark_name]
    assert bench_cfg["type"] == "openml_suite"

    if "suite_id" in bench_cfg:
        suite = openml.study.get_suite(bench_cfg["suite_id"])
    else:
        suite = openml.study.get_suite(bench_cfg["suite_name"])


    extract_runs(
        task_ids=suite.tasks,
        path_csv = csv_path,
        n_top=1,
        metric= "f_measure")

# Uso
export_benchmark_to_csv("openml_cc18", "openml_cc18_meta.csv")
