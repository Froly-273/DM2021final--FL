import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------
# word2vec的skip-gram网络结构，参考自
# https://github.com/Adoni/word2vec_pytorch/blob/master
# ------------------------------------------

class SkipGram(nn.Module):
    def __init__(self, embedding_size, embedding_dim=100):
        super(SkipGram, self).__init__()
        # 定义网络结构
        self.emb_size = embedding_size
        self.emb_dim = embedding_dim
        self.u_embeddings = nn.Embedding(embedding_size, embedding_dim, sparse=True)
        self.v_embeddings = nn.Embedding(embedding_size, embedding_dim, sparse=True)

        # 初始化
        init_range = 0.5 / embedding_dim
        self.u_embeddings.weight.data.uniform_(-init_range, init_range)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, pos_u, pos_v, neg_v):
        """
        u代表一对词中的第一个词，v代表第二个词，pos代表正样本，neg代表经过负采样后的负样本
        u, v不一定在句子中有真实的先后关系，比如一个句子是1234567，那么(u,v)可以是(4,0)也可以是(4,7)
        """

        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        score = torch.mul(emb_u, emb_v).squeeze()
        score = torch.sum(score, dim=1)
        score = F.logsigmoid(score)
        neg_emb_v = self.v_embeddings(neg_v)
        neg_score = torch.bmm(neg_emb_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = F.logsigmoid(-1 * neg_score)
        return -1 * (torch.sum(score)+torch.sum(neg_score))