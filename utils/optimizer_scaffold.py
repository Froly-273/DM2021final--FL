from torch.optim import Optimizer
import torch


class SCAFFOLDOptimizer(Optimizer):
    def __init__(self, params, lr, weight_decay=0):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(SCAFFOLDOptimizer, self).__init__(params, defaults)
        pass

    def step(self, server_controls, client_controls, closure=None):
        loss = None
        if closure is not None:
            loss = closure

        for group, c, ci in zip(self.param_groups, server_controls, client_controls):
            for p in group['params']:
                if p.grad is None:
                    continue
                # d_p = p.grad.data
                # client每一步更新的那个修正量被合并到optimizer里了
                d_p = c.data - ci.data + p.grad.data    # add(dense, sparse)被pytorch支持，add(sparse, dense)不行
                p.data -= group['lr'] * d_p.data       # p = p - lr * dp

        return loss
