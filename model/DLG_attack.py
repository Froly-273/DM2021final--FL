import torch
import numpy as np

from utils.wordlist_input import InputData
from model.skipgram import SkipGram


def DLG_infer(model_index=0, num_clients=10, batch_size=50, embedding_dim=100, use_GPU=True, num_epoch=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 如果model_index用来区分是哪个client的模型（或客户端模型）
    # 目前暂不支持多个模型同时泄露，你可以在外面写一个包装函数，传一个数组进来，对数组里的每个数调用DLG_infer
    assert model_index in range(-1, num_clients)
    if model_index == -1:
        model_name = 'server'
        weight = torch.load('logs/FedAvg/server_best_model.pth')
    else:
        model_name = 'client_%d' % model_index
        weight = torch.load('logs/FedAvg/client_best_model_%s.pth' % str(model_index))
    embedding_size = 16714  # 跟训练集保持一致
    model = SkipGram(embedding_size, embedding_dim)
    model.load_state_dict(weight)
    model.to(device)
    model.eval()
    print("start attacking model %s" % model_name)

    # 制作测试集
    data = InputData(file_name='logs/random_walk_trace_test.txt', min_count=10, client_idx=0, num_clients=10)

    pair_count = data.evaluate_pair_count(window_size=5)  # 返回一个数
    # print(pair_count)   # 10028430
    batch_count = pair_count / batch_size
    # print(batch_count)  # 2005686
    losses = []

    for i in range(int(batch_count)):
        # 把节点按照滑窗的方式建成对，并且只取前batch_size个元素
        pos_pairs = data.get_batch_pairs(batch_size, window_size=5)

        # 正样本 = 上下文 + 中间那个词
        # 负样本（使用负采样获得）= 上下文 + 反正不是中间的任意一个词
        neg_v = data.get_neg_v_neg_sampling(pos_pairs, 5)
        pos_u = [pair[0] for pair in pos_pairs]
        pos_v = [pair[1] for pair in pos_pairs]
        pos_u = torch.autograd.Variable(torch.LongTensor(pos_u))
        pos_v = torch.autograd.Variable(torch.LongTensor(pos_v))
        neg_v = torch.autograd.Variable(torch.LongTensor(neg_v))
        # print(pos_u.shape, pos_v.shape, neg_v.shape)
        if use_GPU:
            pos_u = pos_u.cuda()
            pos_v = pos_v.cuda()
            neg_v = neg_v.cuda()

        with torch.no_grad():
            loss = model.forward(pos_u, pos_v, neg_v)  # loss是他自己还行
        losses.append(loss / batch_size)

    # DLG从这里开始
    # 首先，获取真实梯度
    gt_loss = sum(losses) / len(losses)     # 平均loss
    print("loss on ground truth test set: %.4f" % gt_loss)

    # 我操了，这几句到底怎么写啊
    # gt_loss = torch.tensor(gt_loss, requires_grad=True)
    # dy_dx = torch.autograd.grad(gt_loss, model.parameters(), allow_unused=True)
    dy_dx = torch.autograd.grad(gt_loss, model.parameters())
    print("ground truth gradient", dy_dx)
    original_dy_dx = list((g.detach().clone() for g in dy_dx))

    # 随机生成虚拟输入x和输出y
    # 在word2vec中，由于loss就是model(x)，所以我猜是不需要生成虚拟输出y的
    # 最大的节点编号16862，randint不包含上界所以得写16863
    dummy_data = torch.tensor(np.random.randint(0, 16863, size=10)).to(device).requires_grad_(True)

    # 优化器选择LBFGS
    optimizer = torch.optim.LBFGS([dummy_data])

    best_loss = 100.0
    for epoch in range(num_epoch):
        # 像LBFGS这样的优化器，在step一次的过程中需要多次重复计算，所以得给它定义一个闭包(closure)帮它重复计算
        def closure():
            optimizer.zero_grad()

            dummy_loss = model(dummy_data)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

            grad_diff = 0.0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()

            return grad_diff

        optimizer.step(closure)

        current_loss = closure()
        print("epoch %d, training loss: %.4f" % (epoch+1, current_loss))


