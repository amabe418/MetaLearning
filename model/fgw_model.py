import copy
import logging as log

import numpy as np
import torch
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler

from model.torch_metrics import fgw, distance_matrix
from model.utils import get_ndcg_score


def train_fused_gromov_wasserstein(
    basic_representations,
    pairwise_dist_z,
    learning_rate,
    seed,
    intrinsic_dim,
    early_stopping,
    early_stopping_criterion_ndcg,
    alpha,
    lambda_reg,
    device,
    list_ids,
):
    """
    Entrena una proyección lineal de meta-características básicas usando
    Fused Gromov-Wasserstein (FGW).

    El objetivo es aprender una transformación que proyecte las
    meta-características básicas a un espacio intrínseco de baja dimensión,
    preservando:
      - la estructura de similitud entre datasets (mediante FGW)
      - la calidad del vecindario (medida con NDCG)

    Parámetros
    ----------
    basic_representations : pandas.DataFrame
        Matriz de meta-características básicas.
        Cada fila corresponde a una muestra bootstrap y está indexada
        por el id del dataset.

    pairwise_dist_z : np.ndarray
        Matriz de distancias entre datasets en el espacio objetivo
        (similitud de referencia).

    learning_rate : float
        Tasa de aprendizaje del optimizador Adam.

    seed : int
        Semilla aleatoria para reproducibilidad.

    intrinsic_dim : int
        Dimensión del espacio intrínseco aprendido.

    early_stopping : int
        Número máximo de épocas sin mejora antes de detener el entrenamiento.

    early_stopping_criterion_ndcg : int
        Valor k para calcular NDCG@k durante early stopping.

    alpha : float
        Parámetro de balance entre estructura y características en FGW.

    lambda_reg : float
        Coeficiente de regularización L1 sobre los pesos del modelo.

    device : torch.device
        Dispositivo de ejecución (CPU o GPU).

    list_ids : list
        Lista de identificadores de datasets usados en entrenamiento.

    Retorna
    -------
    best_model : torch.nn.Linear
        Modelo con el mejor NDCG obtenido durante el entrenamiento.

    mds : sklearn.manifold.MDS
        Objeto MDS ajustado para construir el embedding objetivo.
    """
    
    # ---------------------------------------------------------
    # Asociar cada dataset con sus índices de muestras bootstrap
    # ---------------------------------------------------------
    id_reprs = {
        id: np.where(basic_representations.index == id)[0] 
        for id in list_ids
    }

    # Fijar semilla para reproducibilidad
    torch.manual_seed(seed)
    m = len(list_ids) # numero de datasets

    # dimension de entrada (135)
    dim_in = basic_representations.shape[1]

    # ---------------------------------------------------------
    # Paso 1: Construir el espacio objetivo usando MDS
    # ---------------------------------------------------------
    # MDS proyecta los datasets a un espacio de baja dimensión
    # preservando las distancias dadas
    mds = MDS(
        n_components=intrinsic_dim, 
        random_state=seed, 
        dissimilarity="precomputed"
    )

    # Normalizar el embedding obtenido por MDS
    U_ = StandardScaler().fit_transform(
        mds.fit_transform(pairwise_dist_z))

    # ---------------------------------------------------------
    # Paso 2: Preparar tensores de PyTorch
    # ---------------------------------------------------------
    X = torch.from_numpy(
        basic_representations.values
    ).float().to(device)

    U = torch.from_numpy(U_).float().to(device)
    
    # ---------------------------------------------------------
    # Construcción de la matriz estructural M
    # ---------------------------------------------------------
    # M fuerza correspondencia entre el mismo dataset
    # (matriz identidad en esencia)
    M = np.zeros((m, m)) / m
    M[pairwise_dist_z.argsort().argsort() == 0] = 1
    M = torch.from_numpy(M).to(device).float()


    # ---------------------------------------------------------
    # Paso 3: Definir modelo y optimizador
    # ---------------------------------------------------------
    # Proyección lineal desde meta-features básicas
    # al espacio intrínseco
    model = torch.nn.Linear(dim_in, intrinsic_dim, bias=False).to(device).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    ids = range(len(list_ids))


     # ---------------------------------------------------------
    # Variables para early stopping
    # ---------------------------------------------------------
    i = 0
    best_i = i
    best_ndcg = 0
    best_model = copy.deepcopy(model)
    no_improvement = 0

    # ---------------------------------------------------------
    # Bucle de entrenamiento
    # ---------------------------------------------------------
    while no_improvement <= early_stopping:

        optimizer.zero_grad()

        # Seleccionar una muestra bootstrap por dataset
        x_train = X[[np.random.choice(id_reprs[list_ids[_]]) for _ in ids]]
        U_train = U[ids]

        assert not torch.isnan(x_train).any()
        assert not torch.isnan(U_train).any()
        assert not torch.isnan(M).any()

        # Funcion de perdida: FGW + regularizacion L1 
        loss = fgw(
            source=model(x_train),
            target=U_train, 
            device=device, 
            alpha=alpha, 
            M=M
        ) + lambda_reg * torch.norm(model.weight, 1)
        
        assert not torch.isnan(loss).any()

        # Backpropagation
        loss.backward()
        optimizer.step()


        # evaluacion con NDCG
        with torch.no_grad():

            # re-muestreo de instancias bootstrap
            x_train = X[[np.random.choice(id_reprs[list_ids[id]]) for id in ids]]
            
            
            # Distancias predichas en el espacio aprendido
            U_pred_train = distance_matrix(
                pts_src=model(x_train), 
                pts_dst=model(x_train)
            )

            # Distancias reales en el espacio objetivo
            dist_train = distance_matrix(
                pts_src=U, 
                pts_dst=U)

            train_ndcg_score = get_ndcg_score(
                dist_pred=U_pred_train.detach().cpu().numpy(),
                dist_true=dist_train.detach().cpu().numpy(),
                k=early_stopping_criterion_ndcg,
            )


        # criterio de paradas
        if best_ndcg < train_ndcg_score:
            best_i = i
            best_ndcg = train_ndcg_score
            best_model = copy.deepcopy(model)
            no_improvement = 0
        else:
            no_improvement += 1

        loss = loss.item()
        log.info(
            "Epoch {}; train loss: {:.2f}; NDCG@{}: {:.2f}".format(
                i, loss, early_stopping_criterion_ndcg, train_ndcg_score
            )
        )
        i += 1

    log.info(
        "Total epoch: {} -- [BEST NDCG@{}: {:.2f} (Epoch: {})]".format(
            i, early_stopping_criterion_ndcg, best_ndcg, best_i
        )
    )

    return best_model, mds
