import torch

from model.skipgram import SkipGram
from utils.wordlist_input import InputData


class Client():

    def __init__(self, weight, embedding_size, fname='logs/random_walk_trace.txt', initial_lr=0.025, num_epoch=10, batch_size=50, client_idx=0, num_clients=10):
        self.data = InputData(fname, client_idx=client_idx, num_clients=num_clients)
        self.lr = initial_lr
        self.embedding_size = embedding_size
        self.num_epoch =num_epoch
        self.batch_size = batch_size
        self.param = weight
        self.idx = client_idx

    def train(self, embedding_dim=100, use_GPU=True, window_size=5):
        model = SkipGram(self.embedding_size, embedding_dim)
        model.load_state_dict(self.param)       # 不知道这个行不行，一般都是load一个文件地址的
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)

        if use_GPU:
            model.cuda()

        pair_count = self.data.evaluate_pair_count(window_size)  # 返回一个数
        batch_count = pair_count / self.batch_size

        epoch_loss = []
        for epoch in range(self.num_epoch):
            batch_loss = []
            for i in range(int(batch_count)):
                # 把节点按照滑窗的方式建成对，并且只取前batch_size个元素
                pos_pairs = self.data.get_batch_pairs(self.batch_size, window_size)

                # 正样本 = 上下文 + 中间那个词
                # 负样本（使用负采样获得）= 上下文 + 反正不是中间的任意一个词
                neg_v = self.data.get_neg_v_neg_sampling(pos_pairs, 5)
                pos_u = [pair[0] for pair in pos_pairs]
                pos_v = [pair[1] for pair in pos_pairs]
                pos_u = torch.autograd.Variable(torch.LongTensor(pos_u))
                pos_v = torch.autograd.Variable(torch.LongTensor(pos_v))
                neg_v = torch.autograd.Variable(torch.LongTensor(neg_v))
                if use_GPU:
                    pos_u = pos_u.cuda()
                    pos_v = pos_v.cuda()
                    neg_v = neg_v.cuda()

                # 单步训练
                optimizer.zero_grad()
                loss = model.forward(pos_u, pos_v, neg_v)  # loss是他自己还行
                loss.backward()
                optimizer.step()

                # 每10w个batch输出一次结果，它总共有1604544个batch
                if i % 100000 == 0 and i > 0:
                    print("client %d: batch %d in %d finished" % (self.idx, i, int(batch_count)))
                batch_loss.append(loss.item())

            # 每个epoch做一次lr decay，线性的
            lr = self.lr * (1.0 - 1.0 * epoch / self.num_epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            print("client %d: epoch %d in %d finished, training loss: %.4f" % (self.idx, epoch+1, self.num_epoch, sum(epoch_loss) / len(epoch_loss) / self.batch_size))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss) / self.batch_size

