import copy
import torch
import numpy as np
from collections import OrderedDict

from utils.wordlist_input import InputData
from model.skipgram import SkipGram
from model.FedAvg import FedAvg
from model.client import Client
from utils.save_node2vec import save_node2vec_result


def fed_train(fname='logs/random_walk_trace.txt', num_epoch=10, num_client_epoch=10, batch_size=50, initial_lr=0.025, embedding_dim=100,
        num_clients=10, choose_ratio=0.5, use_buffer=0, mix_ratio=0.5, save_folder='logs/FedAvg', seed_clients=None, iid=1, length_seeds=None, iid_real=1):
    print("start training word2vec with FedAvg algorithm")
    print("---------------------------------------------")
    print("parameters:\n"
          "batch size = %d\n"
          "learning rate = %.4f\n"
          "C = %.1f\n"
          "K = %d\n"
          "global epoch = %d\n"
          "client epoch = %d\n"
          "train with weight buffer = %s\n"
          "mix ratio = %.1f\n"
          "identical client dataset size = %s\n"
          "iid distribution = %s"
          % (batch_size, initial_lr, choose_ratio, num_clients, num_epoch, num_client_epoch, bool(use_buffer), mix_ratio, bool(iid & iid_real), bool(iid_real)))
    print("---------------------------------------------")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = InputData(fname)
    embedding_size = len(data.word2id)  # 词典的长度,5254

    model = SkipGram(embedding_size, embedding_dim)  # 词典的长度
    model.to(device)
    global_weight = model.state_dict()     # server的权重，要copy到每个client上保证它们的初始权重一致
    local_weight = [global_weight for i in range(num_clients)]

    loss_epoch = []
    best_loss = 1000.0

    # 保存每个client曾经被选中训练的历史
    # 不考虑文件一致性，没有LFS（日志文件系统）。也就是说，一旦这个buffer里记录了哪些clients将被train，它们一定会被train完，不考虑中途被用户打断的情况
    if use_buffer:
        buffer = [0 for i in range(num_clients)]
        weight_buffer = [copy.deepcopy(global_weight)]   # 记录权重，在buffer FedAvg里有用

    for epoch in range(num_epoch):
        # np.random.seed(epoch+1)
        loss_locals = []
        # 从所有clients中选择C%进行训练
        client_ids = [i for i in range(0, num_clients)]
        m = max(int(choose_ratio * num_clients), 1)  # m = max(C*K, 1)
        if seed_clients is None:
            idxs_users = np.random.choice(client_ids, m, replace=False)  # 随机选m个client进行训练
        else:
            idxs_users = seed_clients[epoch]
        print("\nin epoch %d, client(s) chosen:" % (epoch+1), idxs_users)

        for idx in idxs_users:
            if use_buffer:
                # 根据这个client曾被训练过的次数来决定此次训练的权重
                # -----------------------------------------------
                # 这是最基本的一种模型混合方法（不混合），train过几次就选择第几个epoch的server权重
                if mix_ratio == 1:
                    mix_weight = weight_buffer[buffer[idx]]
                    buffer[idx] += 1    # 记录这个client被使用过了

                # 默认情况
                elif mix_ratio == 0:
                    mix_weight = global_weight

                # 这是另一种方式，各权重按照线性插值混合
                else:
                    mix_weight = OrderedDict()
                    for (k1, v1), (k2, v2) in zip(weight_buffer[buffer[idx]].items(), global_weight.items()):
                        mix_weight[k1] = mix_ratio * v1 + (1 - mix_ratio) * v2
                # -----------------------------------------------
            else:   # 不使用buffer直接做FedAvg
                mix_weight = global_weight
            if length_seeds is not None:
                length_seed = length_seeds[idx]
            else:
                length_seed = None
            if iid_real:
                local = Client(embedding_size=embedding_size, weight=copy.deepcopy(mix_weight), client_idx=idx, num_clients=num_clients,
                               batch_size=batch_size, initial_lr=initial_lr, num_epoch=num_client_epoch, iid=iid, length_seed=length_seed)
            else:
                assert num_clients == 10        # 目前只支持10个client的情况，因为要手动切割数据集，程序做不了
                fname = 'data/noniid/client'+str(idx)+'_walk.txt'
                local = Client(fname=fname, embedding_size=embedding_size, weight=copy.deepcopy(mix_weight), client_idx=idx, num_clients=0,
                               batch_size=batch_size, initial_lr=initial_lr, num_epoch=num_client_epoch, iid=iid, length_seed=length_seed)
                print("client %d got non-iid training data %s" %(idx, fname))
            w, loss = local.train(embedding_dim=embedding_dim)

            local_weight[idx] = copy.deepcopy(w)
            loss_locals.append(loss)

        # 使用FedAvg更新server的weight
        global_weight = FedAvg(local_weight, idxs_users)
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
            embedding = model.u_embeddings.weight.cpu().data.numpy()
            save_node2vec_result(data.id2word, embedding, embedding_dim, output_dir=save_folder + '/word2vec_best.txt')

        # 保存这个epoch的server权重
        if use_buffer:
            weight_buffer.append(copy.deepcopy(global_weight))

    # 保存每个client的模型权重
    for i in range(num_clients):
        torch.save(local_weight[i], save_folder+'/client_model_'+str(i)+'.pth')
    # 保存server的模型权重
    torch.save(global_weight, save_folder+'/server_model.pth')
    # 保存word2vec结果
    embedding = model.u_embeddings.weight.cpu().data.numpy()
    save_node2vec_result(data.id2word, embedding, embedding_dim, output_dir=save_folder+'/word2vec.txt')

