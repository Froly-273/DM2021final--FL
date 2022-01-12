import torch
import os
import shutil

from utils.wordlist_input import InputData
from model.skipgram import SkipGram
from utils.save_node2vec import save_node2vec_result


def train(fname, min_count=10, num_epoch=10, batch_size=50, window_size=5, initial_lr=0.025, embedding_dim=100, use_GPU=True):
    print("Start training word2vec network")
    # 获取字典
    data = InputData(fname, min_count)

    embedding_size = len(data.word2id)  # 词典的长度,5254
    # print(embedding_size)   # 16714
    model = SkipGram(embedding_size, embedding_dim)  # 词典的长度
    if use_GPU:
        model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr)

    pair_count = data.evaluate_pair_count(window_size)  # 返回一个数
    # print(pair_count)   # 100284030
    batch_count = pair_count / batch_size
    # print(batch_count)      # 100284

    epoch_loss = []
    for epoch in range(num_epoch):
        batch_loss = []
        for i in range(int(batch_count)):
            # 把节点按照滑窗的方式建成对，并且只取前batch_size个元素
            pos_pairs = data.get_batch_pairs(batch_size, window_size)

            # 正样本 = 上下文 + 中间那个词
            # 负样本（使用负采样获得）= 上下文 + 反正不是中间的任意一个词
            neg_v = data.get_neg_v_neg_sampling(pos_pairs, 5)
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

            # # 每10w/batch_size个样本做一次lr decay
            # if i * batch_size % 100000 == 0:
            #     lr = initial_lr * (1.0 - 1.0 * i / batch_count)
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = lr

            # 每10w个batch输出一次结果，它总共有1604544个batch
            if i % 100000 == 0 and i > 0:
                print("batch %d in %d finished" % (i, int(batch_count)))
            batch_loss.append(loss.item() / batch_size)

        # 每个epoch做一次lr decay，线性的
        lr = initial_lr * (1.0 - 1.0 * epoch / num_epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        epoch_loss.append(sum(batch_loss) / len(batch_loss))
        print("epoch %d in %d finished, loss: %.3f" % (epoch+1, num_epoch, sum(epoch_loss) / len(epoch_loss)))

    # 保存embedding结果
    if use_GPU:
        embedding = model.u_embeddings.weight.cpu().data.numpy()
    else:
        embedding = model.u_embeddings.weight.data.numpy()

    save_folder = 'logs/baseline'
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    else:
        # 我干的
        # 如果储存模型权重的文件夹还在，那就直接删掉
        # os.remove(save_folder)    # 用os.remove删除非空文件夹时会拒绝访问，用另一个
        shutil.rmtree(save_folder)
        os.makedirs(save_folder)
    torch.save(model.state_dict(), save_folder+'/baseline_model.pth')
    print("End of word2vec network training, total training loss: %.4f" % (sum(epoch_loss) / len(epoch_loss)))
    # return data.id2word, embedding, embedding_dim
    save_node2vec_result(data.id2word, embedding, embedding_dim, output_dir=save_folder+'/word2vec.txt')


def test():
    fname = "../logs/random_walk_trace.txt"
    train(fname, min_count=10, num_epoch=10, batch_size=50, window_size=5, initial_lr=0.025, embedding_dim=100,
          use_GPU=True)


if __name__ == '__main__':
    test()
