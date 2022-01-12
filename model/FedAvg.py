import copy
import torch


def FedAvg(w, idxs):
    """
    输入：w是每个client上的权重
    """
    w_avg = copy.deepcopy(w[idxs[0]])
    for k in w_avg.keys():
        # for i in range(1, len(w)):
        for i in idxs[1:]:
            w_avg[k] += w[i][k]
        # w_avg[k] = torch.div(w_avg[k], len(w))
        w_avg[k] = torch.div(w_avg[k], len(idxs))

    return w_avg
