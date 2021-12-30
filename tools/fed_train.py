import copy
import torch
import os
import numpy as np
import shutil

from utils.wordlist_input import InputData
from model.skipgram import SkipGram
from model.FedAvg import FedAvg
from model.client import Client
from utils.save_node2vec import save_node2vec_result


def fed_train(fname='logs/random_walk_trace.txt', num_epoch=10, num_client_epoch=10, batch_size=50, initial_lr=0.025, embedding_dim=100,
        num_clients=10, choose_ratio=0.5):
    print("start training word2vec with FedAvg algorithm")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = InputData(fname)
    embedding_size = len(data.word2id)  # 词典的长度,5254

    model = SkipGram(embedding_size, embedding_dim)  # 词典的长度
    model.to(device)
    global_weight = model.state_dict()     # server的权重，要copy到每个client上保证它们的初始权重一致
    local_weight = [global_weight for i in range(num_clients)]

    loss_epoch = []
    best_loss = 1000.0

    save_folder = 'logs/FedAvg'
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    else:
        # 我干的
        # 如果储存模型权重的文件夹还在，那就直接删掉
        # os.remove(save_folder)    # 用os.remove删除非空文件夹时会拒绝访问，用另一个
        shutil.rmtree(save_folder)
        os.makedirs(save_folder)

    for epoch in range(num_epoch):
        loss_locals = []
        # 从所有clients中选择C%进行训练
        client_ids = [i for i in range(0, num_clients)]
        m = max(int(choose_ratio * num_clients), 1)  # m = max(C*K, 1)
        idxs_users = np.random.choice(client_ids, m, replace=False)  # 随机选m个client进行训练
        print("\nin epoch %d, client(s) chosen:" % (epoch+1), idxs_users)

        for idx in idxs_users:
            local = Client(embedding_size=embedding_size, weight=copy.deepcopy(global_weight), client_idx=idx, num_clients=num_clients,
                           batch_size=batch_size, initial_lr=initial_lr, num_epoch=num_client_epoch)
            w, loss = local.train(embedding_dim=embedding_dim)

            local_weight[idx] = copy.deepcopy(w)
            loss_locals.append(loss)

        # 使用FedAvg更新server的weight
        global_weight = FedAvg(local_weight)
        model.load_state_dict(global_weight)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('epoch %d, average loss %.4f' % (epoch+1, loss_avg))
        loss_epoch.append(loss_avg)

        # 保存最佳的模型权重，防止过拟合
        if loss_avg < best_loss:
            best_loss = loss_avg
            torch.save(global_weight, save_folder + '/server_best_model.pth')
            for i in range(num_clients):
                torch.save(local_weight[i], save_folder + '/client_best_model_' + str(i) + '.pth')

    # 保存每个client的模型权重
    for i in range(num_clients):
        torch.save(local_weight[i], save_folder+'/client_model_'+str(i)+'.pth')
    # 保存server的模型权重
    torch.save(global_weight, save_folder+'/server_model.pth')
    # 保存word2vec结果
    embedding = model.u_embeddings.weight.cpu().data.numpy()
    save_node2vec_result(data.id2word, embedding, embedding_dim, output_dir=save_folder+'/word2vec.txt')

