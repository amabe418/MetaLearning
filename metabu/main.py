import hydra
import numpy as np

from experiments.tasks import run_task1, run_task2, return_neighbors

@hydra.main(config_path="conf", config_name="config")
def my_app(cfg) -> None:
    np.random.seed(cfg.seed)
    if cfg.task.name == "task1":
        run_task1(cfg)
    elif cfg.task.name == "task2":
        run_task2(cfg)
    elif cfg.task.name == "neighbor":
        return_neighbors(cfg)

if __name__ == "__main__":
    my_app()
