
from config import benchmarks
import openml

# suites = openml.study.list_suites()
# for sid, info in suites.items():
#     print(sid, info['name'])

for bc in benchmarks:
    print(bc)
    bench_cfg = benchmarks[bc]

    assert bench_cfg["type"] == "openml_suite"

    if "suite_id" in bench_cfg:
        suite = openml.study.get_suite(bench_cfg["suite_id"])
    else:
        suite = openml.study.get_suite(bench_cfg["suite_name"])
        
    print("Suite name:", suite.name)
    print("Datasets:", sorted(suite.data))
    print("Tasks:", len(suite.tasks))

    if "expected_num_tasks" in bench_cfg:
        assert len(suite.tasks) == bench_cfg["expected_num_tasks"]
