import argparse
import os
import time
import numpy as np
import shutil

from tools.train import train
from tools.generate_walk import generate_random_walk
from tools.similarity_predict import evaluate
from utils.score import AUCscore

# ---------------------------------------------------------------------
# 这两个东西，如果你想选择多线程并发训练的话，选下面一个，否则选上面一个
# 不用担心，两个的API我写的完全一样，只需要改这一行就可以完美兼容了
# 多线程的话，CPU和GPU利用率会提高那么一点
# 但是由于python的互斥锁没有C那么先进，大部分时间花在spin上，所以速度其实没有快多少，2.5小时到2小时左右吧

# from tools.fed_train import fed_train
# from tools.fed_train_multithread import fed_train
# ---------------------------------------------------------------------


def make_parser():
    parser = argparse.ArgumentParser("Argument parser for federated learning demo.")
    # 模式选择：w=生成随机游走路径，o=普通训练，f=FedAvg训练，s=SCAFFOLD训练
    parser.add_argument("-m", "--mode", type=str, default="f", help="mode of this demo: w=generate random walk, o=ordinary training, f=FedAvg training, s=SCAFFOLD training")
    # batch size
    parser.add_argument("-b", "--batch_size", type=int, default=10000, help="batch size")
    # 初始lr
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.025, help="learning rate")
    # word2vec的嵌入词向量维度
    parser.add_argument("-dim", "--embedding_dim", type=int, default=100, help="embedding dim for word2vec")
    # 选择client的随机数种子
    parser.add_argument("-s", "--seed", type=int, default=None, help="training seed, useful to deterministically choose clients")

    # --------------------------------
    # 生成随机游走的参数
    # --------------------------------
    # p
    parser.add_argument("-p", "--prob_out", type=float, default=1.0, help="p value of node2vec sampling")
    # q
    parser.add_argument("-q", "--prob_return", type=float, default=1.0, help="q value of node2vec sampling")
    # 对于每个节点，生成多少条以它为起点的随机游走路径
    parser.add_argument("-ew", "--epoch_walk", type=int, default=100, help="number of iterations of node2vec sampling")

    # --------------------------------
    # 普通训练的参数
    # --------------------------------
    # epoch
    parser.add_argument("-e", "--epoch", type=int, default=1, help="epoch of ordinary training")

    # --------------------------------
    # FL训练的参数
    # --------------------------------
    # 数据分布是否iid，这主要是由每个client分到的数据集规模不同决定的，seed同样可以对这里生效
    parser.add_argument("-is", "--iid", type=int, default=1, help="identical dataset size or not")
    # 真实的iid，如果设置了这个参数，每个client会被直接分到不同的数据集
    parser.add_argument("-i", "--iid_real", type=int, default=1, help="iid distribution or not")
    # server epoch
    parser.add_argument("-es", "--epoch_server", type=int, default=5, help="epoch of server in federated learning")
    # client epoch
    parser.add_argument("-ec", "--epoch_client", type=int, default=5, help="epoch of client in federated learning")
    # client数量
    parser.add_argument("-c", "--clients", type=int, default=10, help="number of clients in federated learning")
    # 每一个server epoch中，被选择的client占比
    parser.add_argument("-r", "--choose_ratio", type=float, default=0.4, help="ratio of choosing clients in each server epoch")
    # 多线程训练
    parser.add_argument("-t", "--multithread", type=int, default=0, help="use multi-thread training")
    # 使用权重历史记录（weight buffer）
    parser.add_argument("-buf", "--use_buffer", type=int, default=0, help="use server weight buffer to train")
    # 权重混合的比例
    parser.add_argument("-rmix", "--mix_ratio", type=float, default=0.5, help="mix ratio when weight mixing is activated")

    return parser


if __name__ == "__main__":
    source_path = 'data/lab2_edge.csv'
    test_path = 'data/lab2_test.csv'
    ground_truth_path = 'data/lab2_truth.csv'

    args = make_parser().parse_args()

    now = time.localtime()
    t = time.strftime("%Y%m%d_%H%M%S", now)
    root_dir = 'logs/' + t      # 保存本次实验结果的根目录
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)
    walk_path = root_dir + '/random_walk_trace.txt'

    mode = args.mode
    assert mode in ["w", "o", "f", "s"]

    iid = args.iid

    # 生成随机游走
    if mode == "w":
        p = args.prob_out
        q = args.prob_return
        num_epoch = args.epoch_walk
        generate_random_walk(source_path, walk_path, p, q, num_epoch)
    else:
        batch_size = args.batch_size
        lr = args.learning_rate
        embedding_dim = args.embedding_dim
        seed = args.seed
        num_server_epoch = args.epoch_server
        num_client_epoch = args.epoch_client
        num_clients = args.clients
        C = args.choose_ratio
        iid_real = args.iid_real

        # 随机种子训练，这将导致每个server epoch选择的client完全固定
        if seed is not None:
            print("Warning: You have chosen seed training %d" % seed)
            np.random.seed(seed)
            m = max(int(C * num_clients), 1)
            seed_clients = []
            list_clients = [i for i in range(num_clients)]
            for epoch in range(num_server_epoch):
                c = np.random.choice(list_clients, m, replace=False)
                seed_clients.append(c)
                print("epoch %d, clients:" % epoch, list(c))
        else:
            seed_clients = None

        if not iid and seed is not None:
            # 这是随机数据集长度的种子
            length_seeds = [np.random.rand() for i in range(num_clients)]
            print("length seed of each client's dataset:")
            ls4show = ["%.2f" % i for i in length_seeds]
            print(ls4show)
        else:
            length_seeds = None

        # seed_clients = [[2,9,6,4],[4,1,5,0],[5,4,1,2],[3,8,4,9],[9,5,2,4]]

        if not os.path.isfile(root_dir + '/random_walk_trace.txt'):
            walk_path = 'logs/random_walk_trace.txt'
        else:
            walk_path = root_dir + '/random_walk_trace.txt'

        # 普通训练
        if mode == "o":
            num_epoch = args.epoch
            train(walk_path, num_epoch=num_epoch, batch_size=batch_size, embedding_dim=embedding_dim, initial_lr=lr)
            evaluate(root_dir+'/word2vec.txt', test_path, root_dir+'/predictions.csv')
            print("baseline model accuracy: %.2f%%" % (AUCscore(pred_path=root_dir+'/predictions.csv') * 100))

        # FedAvg训练
        elif mode == "f":
            thread = args.multithread
            # 单线程
            if not thread:
                from tools.fed_train import fed_train
                use_buffer = args.use_buffer
                mix_ratio = args.mix_ratio
                fed_train(walk_path, num_server_epoch, num_client_epoch, batch_size, lr, embedding_dim, num_clients, C, use_buffer, mix_ratio, root_dir, seed_clients, iid, length_seeds, iid_real)
                evaluate(root_dir+'/word2vec.txt', test_path, root_dir+'/predictions.csv')
                print("FedAvg model accuracy: %.2f%%" % (AUCscore(pred_path=root_dir+'/predictions.csv') * 100))
            # 多线程
            else:
                from tools.fed_train_multithread import fed_train

                fed_train(walk_path, num_server_epoch, num_client_epoch, batch_size, lr, embedding_dim, num_clients, C, root_dir, seed_clients, iid, length_seeds, iid_real)
                evaluate(root_dir+'/word2vec.txt', test_path, root_dir+'/predictions.csv')
                print("FedAvg (multi-thread) model accuracy: %.2f%%" % (AUCscore(pred_path=root_dir+'/predictions.csv') * 100))

        # SCAFFOLD训练
        else:
            from tools.scaffold_train import scaffold_train
            num_server_epoch = args.epoch_server
            num_client_epoch = args.epoch_client
            num_clients = args.clients
            C = args.choose_ratio
            scaffold_train(walk_path, num_server_epoch, num_client_epoch, batch_size, lr, embedding_dim, num_clients, C, root_dir, seed_clients, iid, length_seeds, iid_real)
            evaluate(root_dir+'/word2vec.txt', test_path, root_dir+'/predictions.csv')
            print("SCAFFOLD model accuracy: %.2f%%" % (AUCscore(pred_path=root_dir+'/predictions.csv') * 100))




