import numpy as np
import pandas as pd
import torch

import logging as log
import typing

from model.utils import get_cost_matrix, get_pca_importances
from model.distance import intrinsic_estimator , distance_wasserstein
from model.fgw_model import train_fused_gromov_wasserstein

class MetaFeatX:
    def __init__(
        self,
        alpha: float = 0.5,
        lambda_reg: float = 1e-3,
        learning_rate: float = 0.01,
        early_stopping_patience: int = 10,
        early_stopping_criterion_ndcg: int = 10,
        verbose: bool = True,
        pairwise_target_dist_func: typing.Callable = distance_wasserstein,
        ncpus: int = 1,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        
        self.early_stopping_criterion_ndcg = early_stopping_criterion_ndcg
        self.seed = seed
        self.ncpus = ncpus
        self.verbose = verbose
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.model = None
        self.mds = None
        self.intrinsic_dim = None
        self.device = torch.device(device)
        self.pairwise_target_dist_func = pairwise_target_dist_func

        if verbose:
            log.basicConfig(format="%(asctime)s: %(message)s", level=log.DEBUG)
        else:
            log.basicConfig(format="%(asctime)s : %(message)s")

    @property
    def linear_mapping(self) -> np.ndarray:

        return self.model.weight.detach().cpu().numpy()


    def train(
        self, 
        basic_reprs: pd.DataFrame, 
        target_reprs: pd.DataFrame,
        column_id: str
    ) -> None:

        # ordena todas las tareas unicas de basic_reprs
        list_ids = sorted(list(basic_reprs[column_id].unique()))

        # comprueba que todas estas tareas tengan representationes objetivo
        task_id_has_target_representation = target_reprs[column_id].unique()
        if set(list_ids) != set(task_id_has_target_representation):
            raise ValueError("Inconsistent numbers of instances.")

        # se guarda una lista de features basicas
        basic_repr_labels = basic_reprs.columns
        self.basic_repr_labels = [str(_) for _ in basic_repr_labels if _ != column_id]
        log.info(
            "Considering {0} basic meta-features: ".format(len(self.basic_repr_labels))
            + ",".join(self.basic_repr_labels)
        )

        # calculo de la matriz de distancias entre targets
        log.info("Compute pairwise distance of target representations.")
        # print(target_reprs.shape)
        # print(target_reprs)
        # print("Cantidad de task",len(list_ids))

        self.cost_matrix = get_cost_matrix(
            target_repr=target_reprs,
            task_ids=list_ids,
            verbose=self.verbose,
            ncpus=self.ncpus,
        )

        assert self.cost_matrix.shape[0] == len(list_ids)
    
        # print(f"Cost matrix:\n{self.cost_matrix}")

        # estimacion de la dimension intrinseca
        log.info("Compute intrinsic dimension.")
        self.intrinsic_dim = intrinsic_estimator(self.cost_matrix)
        print(f"Intrinsic dimension: {self.intrinsic_dim}")

        # Aprendizaje de la representacion del modelo
        log.info("Train MetaFeatX meta-features.")
        self.model, self.mds = train_fused_gromov_wasserstein(
            basic_representations=basic_reprs.set_index(column_id),
            pairwise_dist_z=self.cost_matrix,
            learning_rate=self.learning_rate,
            seed=self.seed,
            early_stopping=self.early_stopping_patience,
            early_stopping_criterion_ndcg=self.early_stopping_criterion_ndcg,
            alpha=self.alpha,
            intrinsic_dim=self.intrinsic_dim,
            lambda_reg=self.lambda_reg,
            device=self.device,
            list_ids=list_ids,
        )

        print(f"Trained linear mapping shape: {self.linear_mapping.shape}")
        # print(f"Trainerd linear mapping:\n{self.model.shape}")
        return self

    def predict(
        self, 
        basic_reprs: pd.DataFrame
    ) -> np.ndarray:
        """Predict meta-features for new basic representations"""

        return np.dot(basic_reprs[self.basic_repr_labels].values, self.linear_mapping.T)


    def train_and_predict(
            self,
            basic_reprs: pd.DataFrame,
            target_reprs: pd.DataFrame,
            column_id: str,
            train_ids : list,
            test_ids: list
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        
        basic_reprs_train = basic_reprs  # [basic_reprs[column_id].isin(train_ids)]
        basic_reprs_test = basic_reprs[basic_reprs[column_id].isin(test_ids)]
        target_reprs_train = target_reprs  # [target_reprs[column_id].isin(train_ids)]

        self.train(
            basic_reprs = basic_reprs_train,
            target_reprs = target_reprs_train,
            column_id = column_id
        )
        
        return (
            self.predict(basic_reprs_train[basic_reprs[column_id].isin(train_ids)]),
            self.predict(basic_reprs_test),
        )

    def get_features_importances(self) -> typing.Tuple[np.ndarray, typing.List[str]]:
        
        # calcula la importancia usando pca sobre el embedding mds
        imp = get_pca_importances(self.mds.embedding_)
        
        # seleccion la dimension mas importante
        idx_best = imp.argmax()

        # extrae los pesos de la transformacion lineal de esa dimension
        assert len(np.abs(self.linear_mapping[idx_best])) == len(self.basic_repr_labels)
        return np.abs(self.linear_mapping[idx_best]), self.basic_repr_labels

