# import numpy as np
# from scipy.optimize import linear_sum_assignment as hungarian
# from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score

# cluster_nmi = normalized_mutual_info_score
# def cluster_acc(y_true, y_pred):
#     y_true = y_true.astype(np.int64)
#     assert y_pred.size == y_true.size
#     D = max(y_pred.max(), y_true.max()) + 1
#     w = np.zeros((D, D), dtype=np.int64)
#     for i in range(y_pred.size):
#         w[y_pred[i], y_true[i]] += 1
  
#     # ind = sklearn.utils.linear_assignment_.linear_assignment(w.max() - w)
#     # row_ind, col_ind = linear_assignment(w.max() - w)
#     row_ind, col_ind = hungarian(w.max() - w)
#     return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size

import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics import accuracy_score

def cluster_scores(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate NMI and ARI
    nmi_score = normalized_mutual_info_score(y_true, y_pred)
    ari_score = adjusted_rand_score(y_true, y_pred)
    
    return accuracy, nmi_score, ari_score

