import shutil
from pathlib import Path

import numpy as np
import openml
from ConfigSpace.configuration_space import Configuration
from smac.facade.smac_hpo_facade import SMAC4HPO
# Import SMAC-utilities
from smac.scenario.scenario import Scenario

from .openml_pimp import set_up_pipeline_for_task

openml.config.logger.propagate = False
openml.datasets.dataset.logger.propagate = False

from smac.callbacks import IncorporateRunResultCallback


openml.config.logger.propagate = False
openml.datasets.dataset.logger.propagate = False


class ResultCallback(IncorporateRunResultCallback):
    def __init__(self, task_id, pipeline, counter):
        self.task_id, self.counter, self.pipeline = task_id, counter, pipeline
        self.results = []

    def __call__(self, smbo, run_info, result, time_left):
        self.results.append({
            "task_id": self.task_id,
            "pipeline": self.pipeline,
            "hp_id": self.counter,
            "hp": run_info.config.get_dictionary(),
            "status": str(result.status),
            "performance": -result.cost
        })
        self.counter += 1


class PipelineTuner():
    """
    Clase para optimizar hiperparámetros de un pipeline de Machine Learning
    utilizando SMAC (Sequential Model-based Algorithm Configuration) como optimizador.

    Esta clase genera funciones "caja negra" que evalúan un pipeline sobre tareas
    OpenML y administra la ejecución de SMAC, incluyendo inicialización de configuraciones
    y recolección de resultados.
    """

    def __init__(self, pipeline, config_space, seed):
        """
        Clase para optimizar hiperparámetros de un pipeline de Machine Learning
        utilizando SMAC (Sequential Model-based Algorithm Configuration) como optimizador.

        Esta clase genera funciones "caja negra" que evalúan un pipeline sobre tareas
        OpenML y administra la ejecución de SMAC, incluyendo inicialización de configuraciones
        y recolección de resultados.
        """
        self.pipeline = pipeline
        self.config_space = config_space
        self.seed = seed

    def generate_black_box_function(self, task_id, n_jobs):
        """
        Genera una función "caja negra" que evalúa un pipeline de Machine Learning
        en una tarea específica de OpenML. Esta función está diseñada para ser utilizada
        por optimizadores de hiperparámetros como SMAC.

        Args:
            task_id (int): ID de la tarea OpenML que se va a evaluar.
            n_jobs (int): Número de hilos/procesos para paralelizar la evaluación.

        Returns:
            function: Una función que recibe un objeto Configuration (con hiperparámetros)
                    y devuelve una tupla (costo, info_adicional), donde 'costo' es el 
                    valor negativo de la precisión media (para minimizar en SMAC) y 
                    'info_adicional' contiene métricas adicionales.
        """

        def black_box_function(config):
            """
            Función interna que evalúa un pipeline con hiperparámetros específicos
            sobre la tarea OpenML indicada.

            Args:
                config (ConfigSpace.Configuration): Objeto que contiene los hiperparámetros a evaluar.

            Returns:
                tuple: (costo, info_adicional)
                    - costo (float): negativo de la precisión media del pipeline en la tarea.
                    - info_adicional (dict): diccionario para retornar métricas adicionales.
            """
            # Convertir configuración de SMAC a diccionario Python
            cfg = config.get_dictionary()

            # Crear el pipeline específico para la tarea
            pipe = set_up_pipeline_for_task(task_id, self.pipeline)

            # Asignar los hiperparámetros al pipeline
            pipe.set_params(**cfg)

            # Ejecutar el pipeline sobre la tarea OpenML y obtener las evaluaciones
            run = openml.runs.run_model_on_task(
                pipe,
                task_id,
                avoid_duplicate_runs=False,
                dataset_format="array",  # cambiar a 'dataframe' para evitar futuros warnings
                n_jobs=n_jobs
            )

            # Obtener precisión promedio de todos los folds
            mean_accuracy = np.mean(list(run.fold_evaluations['predictive_accuracy'][0].values()))

            # Devolver negativo de la precisión (SMAC minimiza) y diccionario vacío
            return -mean_accuracy, {}

        # Retornar la función "caja negra" lista para ser usada por SMAC
        return black_box_function


    def clean_dict(self, dictionary):
        """
        Limpia un diccionario de hiperparámetros, convirtiendo strings 'True'/'False' en booleanos.

        Args:
            dictionary (dict): Diccionario con valores de hiperparámetros (puede contener strings).

        Returns:
            dict: Diccionario con valores convertidos a tipos correctos.
        """
        c = {"True": True, "False": False}
        return  {k: c[v] if v in c else v for k, v in dictionary.items()}

    def exec(self, task_id, hps, counter):
        """
        Ejecuta la optimización de hiperparámetros para un pipeline en una tarea OpenML.

        Crea configuraciones iniciales, define el escenario SMAC, registra callbacks,
        y ejecuta el optimizador. Al final limpia archivos temporales generados.

        Args:
            task_id (int): ID de la tarea OpenML.
            hps (list of dict): Lista de configuraciones iniciales de hiperparámetros.
            counter (int): Contador para identificar resultados de configuraciones.

        Returns:
            list: Lista de resultados de SMAC, cada uno como diccionario con información
                  de desempeño y configuración de hiperparámetros.
        """

        # Convertir las configuraciones iniciales en objetos Configuration para SMAC
        init_configs = [Configuration(configuration_space=self.config_space, values=self.clean_dict(hp)) for hp in hps]

        # Crear la función "caja negra" que SMAC evaluará
        obective_function = self.generate_black_box_function(task_id=task_id, n_jobs=1)

        # Definir el escenario SMAC
        scenario = Scenario({
            "run_obj": "quality",          # Objetivo: optimizar calidad (accuracy)
            "runcount-limit": len(init_configs),  # Número máximo de evaluaciones
            "cs": self.config_space,       # Espacio de hiperparámetros
            "deterministic": "true",       # Problema determinístico
            "execdir": "/tmp",             # Directorio temporal de SMAC
            "cutoff": 60 * 15,             # Timeout por evaluación (segundos)
            "memory_limit": 5000,          # Límite de memoria (MB)
            "cost_for_crash": 1,           # Penalización por crash
            "abort_on_first_run_crash": False
        })

        # Crear optimizador SMAC
        smac = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(self.seed),
                        tae_runner=obective_function, initial_configurations=init_configs, initial_design=None)

        # Crear callback para recolectar resultados
        cal = ResultCallback(task_id=task_id, pipeline=self.pipeline, counter=counter)
        smac.register_callback(cal)

        # Ejecutar la optimización
        smac.optimize()

        # Limpiar archivos temporales generados por SMAC
        path = Path(smac.output_dir)
        parent = path.parent.absolute()
        shutil.rmtree(parent, ignore_errors=True)

        # Retornar resultados recolectados
        return cal.results

