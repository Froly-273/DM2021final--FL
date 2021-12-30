from tools.train import train
from tools.generate_walk import generate_random_walk
from tools.similarity_predict import evaluate

# ---------------------------------------------------------------------
# 这两个东西，如果你想选择多线程并发训练的话，选下面一个，否则选上面一个
# 不用担心，两个的API我写的完全一样，只需要改这一行就可以完美兼容了
# 多线程的话，CPU和GPU利用率会提高那么一点
# 但是由于python的互斥锁没有C那么先进，大部分时间花在spin上，所以速度其实没有快多少

# from tools.fed_train import fed_train
from tools.fed_train_multithread import fed_train
# ---------------------------------------------------------------------

if __name__ == "__main__":
    source_path = 'data/lab2_edge.csv'
    test_path = 'data/lab2_test.csv'
    walk_path = 'logs/random_walk_trace.txt'

    # -------------------------------------------------------------------
    # 随机游走的参数：返回概率p，出入概率q，每个节点迭代次数num_walk_iterations
    p = 1.0
    q = 1.0
    num_walk_iterations = 100
    # -------------------------------------------------------------------

    # 这一步结束时，logs/baseline里会多出一个random_walk_trace.txt
    generate_random_walk(source_path, walkpath=walk_path, p=p, q=q, num_iterations=num_walk_iterations)

    # -------------------------------------------------------------------
    # word2vec的参数：batch_size，learning_rate，embedding维数，epoch数
    batch_size = 1000        # batch size很大是正常的，毕竟这个数据集就大的离谱，batch size不大的话估计两晚上也train不完
    lr = 0.025
    embedding_dim = 100
    num_epoch = 10
    # -------------------------------------------------------------------

    # 多出一个baseline_model.pth和word2vec.txt
    train(walk_path, num_epoch=num_epoch, batch_size=batch_size, embedding_dim=embedding_dim, initial_lr=lr)
    # 多出一个predictions.csv
    evaluate('logs/baseline/word2vec.txt', test_path, 'logs/baseline/predictions.csv')

    # -------------------------------------------------------------------
    # FedAvg的参数：每个client内部训练的epoch数，clients数量，每个client以多大概率选中
    num_client_epoch = 10
    num_clients = 10
    C = 0.4
    # -------------------------------------------------------------------

    # logs/FedAvg里会多出一堆.pth权重文件
    fed_train(walk_path, num_clients=num_clients, choose_ratio=C, batch_size=batch_size, num_epoch=num_epoch,
              num_client_epoch=num_client_epoch, initial_lr=lr)
    evaluate('logs/FedAvg/word2vec.txt', test_path, 'logs/FedAvg/predictions.csv')
