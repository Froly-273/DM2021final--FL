import torch

from utils.wordlist_input import InputData
from model.skipgram import SkipGram


def DLG_infer(model_index=0, num_clients=10, gt_input_path='data/lab2_test.csv', embedding_dim=100):
    # 如果model_index用来区分是哪个client的模型（或客户端模型）
    # 目前暂不支持多个模型同时泄露，你可以在外面写一个包装函数，传一个数组进来，对数组里的每个数调用DLG_infer
    assert model_index in range(-1, num_clients)
    if model_index == -1:
        weight = torch.load('logs/FedAvg/server_best_model.pth')
    else:
        weight = torch.load('logs/FedAvg/client_best_model_%s.pth' % str(model_index))
