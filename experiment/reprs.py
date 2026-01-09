import pandas as pd
from .utils import load_bootstrap_representations
from model import MetaFeatX
import numpy as np

def load_model_representations(cfg, basic_reprs, target_reprs, list_ids, train_ids, test_ids):
    basic_reprs["boostrap"] = 0

    bootstrap_reprs = load_bootstrap_representations(metafeature=cfg.metafeature, path=cfg.data_path)

    print(bootstrap_reprs.shape)
    bootstrap_reprs = bootstrap_reprs[bootstrap_reprs.task_id.isin(list_ids)]

    
    bootstrap_reprs["boostrap"] = 1

    print(bootstrap_reprs.shape)

    combined_basic_reprs = pd.concat([basic_reprs, bootstrap_reprs], axis=0)

    print(combined_basic_reprs.shape)

    
    combined_basic_reprs = pd.concat([
        combined_basic_reprs[combined_basic_reprs.task_id.isin(train_ids)],
        combined_basic_reprs[combined_basic_reprs.task_id.isin(test_ids)]
    ], axis=0)

    print(combined_basic_reprs.shape)
    

    model = MetaFeatX(alpha=0.5,
                    lambda_reg=1e-3,
                    learning_rate=0.01,
                    early_stopping_patience=20,
                    early_stopping_criterion_ndcg=cfg.task.ndcg,
                    verbose=False,
                    seed=cfg.seed)
    repr_train, repr_test = model.train_and_predict(
        basic_reprs=combined_basic_reprs.drop(["boostrap"], axis=1),
        target_reprs=target_reprs,
        column_id="task_id",
        train_ids=train_ids,
        test_ids=test_ids
    )

    print(repr_train.shape, repr_test.shape)
    print(repr_train)
    print(repr_test)

    model_reprs = np.concatenate([repr_train, repr_test], axis=0)
    print(f"El shape es de :{model_reprs.shape}")
    print(f"model_reprs es :{model_reprs}")
    model_reprs = pd.DataFrame(model_reprs, columns=[f"col{_}" for _ in range(model_reprs.shape[1])])
    print(f"El shape es de :{model_reprs.shape}")
    print(f"model_reprs es :{model_reprs}")
    model_reprs["task_id"] = combined_basic_reprs["task_id"].values
    model_reprs["boostrap"] = combined_basic_reprs["boostrap"].values
    print(f"El shape es de :{model_reprs.shape}")
    print(f"model_reprs es :{model_reprs}")
    return model_reprs[model_reprs.boostrap == 0].drop(["boostrap"], axis=1)
