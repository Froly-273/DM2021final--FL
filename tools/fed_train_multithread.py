import copy
import torch
import numpy as np
import threading

from utils.wordlist_input import InputData
from model.skipgram import SkipGram
from model.FedAvg import FedAvg
from model.client import Client
from utils.save_node2vec import save_node2vec_result


def client_job(embedding_size, global_weight, client_idx, num_clients, batch_size, initial_lr, num_client_epoch, local_weight, loss_locals, embedding_dim=100, iid=1, length_seed=None, iid_real=1):
    if iid_real:
        local = Client(embedding_size=embedding_size, weight=copy.deepcopy(global_weight), client_idx=client_idx, num_clients=num_clients,
                       batch_size=batch_size, initial_lr=initial_lr, num_epoch=num_client_epoch, iid=iid, length_seed=length_seed)
    else:
        assert num_clients == 10  # 目前只支持10个client的情况，因为要手动切割数据集，程序做不了
        fname = 'data/noniid/client' + str(client_idx) + '_walk.txt'
        local = Client(fname=fname, embedding_size=embedding_size, weight=copy.deepcopy(global_weight), client_idx=client_idx, num_clients=0,
                       batch_size=batch_size, initial_lr=initial_lr, num_epoch=num_client_epoch, iid=iid, length_seed=length_seed)
        print("client %d got non-iid training data %s" % (client_idx, fname))
    w, loss = local.train(embedding_dim=embedding_dim)
    local_weight[client_idx] = copy.deepcopy(w)
    # loss_locals.append(loss)
    loss_locals[client_idx] += loss     # 之前那个可能会有多线程竞争，因为哪个线程想用直接添加一个数就行。这个不会，每个的位置都是确定的


def fed_train(fname='logs/random_walk_trace.txt', num_epoch=10, num_client_epoch=10, batch_size=50, initial_lr=0.025, embedding_dim=100,
        num_clients=10, choose_ratio=0.5, save_folder='logs/FedAvgThread', seed_clients=None, iid=1, length_seeds=None, iid_real=1):
    print("start training word2vec with FedAvg algorithm")
    print("---------------------------------------------")
    print("parameters:\n"
          "batch size = %d\n"
          "learning rate = %.4f\n"
          "C = %.1f\n"
          "K = %d\n"
          "global epoch = %d\n"
          "client epoch = %d\n"
          "identical client dataset size = %s\n"
          "iid distribution = %s"
          % (batch_size, initial_lr, choose_ratio, num_clients, num_epoch, num_client_epoch, bool(iid & iid_real), bool(iid_real)))
    print("---------------------------------------------")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = InputData(fname)
    embedding_size = len(data.word2id)  # 词典的长度,5254
    # print(embedding_size)       # 16714

    model = SkipGram(embedding_size, embedding_dim)  # 词典的长度
    model.to(device)
    global_weight = model.state_dict()     # server的权重，要copy到每个client上保证它们的初始权重一致
    local_weight = [global_weight for i in range(num_clients)]

    loss_epoch = []
    best_loss = 1000.0

    # 线程池
    threads = []

    for epoch in range(num_epoch):
        loss_locals = [0.0 for i in range(num_clients)]
        # 从所有clients中选择C%进行训练
        client_ids = [i for i in range(0, num_clients)]
        m = max(int(choose_ratio * num_clients), 1)  # m = max(C*K, 1)
        if seed_clients is None:
            idxs_users = np.random.choice(client_ids, m, replace=False)  # 随机选m个client进行训练
        else:
            idxs_users = seed_clients[epoch]
        print("\nin epoch %d, client(s) chosen:" % (epoch+1), idxs_users)
        print("training with %d threads" % len(idxs_users))

        # 每个client分配一个线程去运行
        for idx in idxs_users:
            if length_seeds is not None:
                length_seed = length_seeds[idx]
            else:
                length_seed = None
            t = threading.Thread(target=client_job, args=(embedding_size, global_weight, idx, num_clients, batch_size, initial_lr,
                                                          num_client_epoch, local_weight, loss_locals, embedding_dim, iid, length_seed, iid_real))
            threads.append(t)
            t.start()

        # 回收线程
        for t in threads:
            t.join()

        # 使用FedAvg更新server的weight
        global_weight = FedAvg(local_weight, idxs_users)
        model.load_state_dict(global_weight)

        # print loss
        # loss_avg = sum(loss_locals) / len(loss_locals) # 这个不能再这么算了，len(loss_locals)固定等于10
        loss_avg = sum(loss_locals) / m
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

    # 保存每个client的模型权重
    for i in range(num_clients):
        torch.save(local_weight[i], save_folder+'/client_model_'+str(i)+'.pth')
    # 保存server的模型权重
    torch.save(global_weight, save_folder+'/server_model.pth')
    # 保存word2vec结果
    embedding = model.u_embeddings.weight.cpu().data.numpy()
    save_node2vec_result(data.id2word, embedding, embedding_dim, output_dir=save_folder+'/word2vec.txt')

