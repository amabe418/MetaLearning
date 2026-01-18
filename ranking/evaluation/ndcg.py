from sklearn.metrics import ndcg_score
import numpy as np

def get_ndcg_score(dist_pred, dist_true, k=10):
    
    pred_rank = dist_pred.argsort().argsort()
    true_rank = dist_true.argsort().argsort()

    pred_rank[np.where(pred_rank < k)] = 1
    pred_rank[np.where(pred_rank >= k)] = 0
    true_rank[np.where(true_rank < k)] = 1
    true_rank[np.where(true_rank >= k)] = 0

    return ndcg_score(y_true=true_rank, y_score=pred_rank, k=k)