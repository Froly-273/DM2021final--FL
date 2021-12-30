import numpy as np
import random
from utils.alias import alias_setup, alias_sampling


class RandomWalk():

    def __init__(self, G, p=1.0, q=1.0):
        self.graph = G  # networkx类的图
        self.p = p      # 返回概率p
        self.q = q      # 出入概率q

    def preprocess(self):
        """
            对每一对节点，计算它的权重

            :param graph: 一个Networkx有向图
            :param p: 返回概率p
            :param q: 出入概率q
            :return: 每个节点邻居的权重alias_nodes，每对边的权重alias_edges
            """

        alias_nodes = {}
        for node in self.graph.nodes():
            # 如果两个节点相连，设置权重为1，否则为0
            nbr_probs = [self.graph[node][nbr].get('weight', 1.0) for nbr in self.graph.neighbors(node)]
            # partition function Z
            Z = sum(nbr_probs)

            # 构建普朗克能量，即Pr(i,j) = exp(i,j) / Z
            nbr_probs_normalized = [float(p) / Z for p in nbr_probs]
            # 解决非均匀采样问题
            alias_nodes[node] = alias_setup(nbr_probs_normalized)

        alias_edges = {}
        for edge in self.graph.edges():
            edge_probs = []
            # 这里t,v,x的记号与论文中的图表示一致，假设已经采样了(t,v)，要确定下一个节点x
            t = edge[0]
            v = edge[1]

            for x in sorted(self.graph.neighbors(v)):
                # t=x，概率设置为1/p
                weight = self.graph[v][x].get('weight', 1.0)
                if t == x:
                    edge_probs.append(weight / self.p)
                # d(t,x)=1，概率设置为1
                elif self.graph.has_edge(t, x):
                    edge_probs.append(weight)
                # d(t,x)>=2，概率设置为1/q
                else:
                    edge_probs.append(weight / self.q)
            # partition funtion Z
            Z = sum(edge_probs)
            # 构建普朗克能量，即Pr(i,j) = exp(i,j) / Z
            edge_probs_normalized = [float(p) / Z for p in edge_probs]
            alias_edges[edge] = alias_setup(edge_probs_normalized)

        print("Preprocess finished")
        # return alias_nodes, alias_edges
        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

    def get_one_chain(self, start_node, chain_length=10):
        """
        生成从节点start_node开始的随机游走链，默认链长度为10
        """

        walk = [start_node]
        while len(walk) < chain_length:
            cur = walk[-1]
            cur_nbrs = sorted(self.graph.neighbors(cur))
            if len(cur_nbrs) > 0:
                # 初始的时候只能从这个点cur出发，查找cur_nbrs
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_sampling(self.alias_nodes[cur][0], self.alias_nodes[cur][1])])
                # 之后可以查找三连，即prev, cur, cur_nbrs
                else:
                    prev = walk[-2]
                    next = cur_nbrs[alias_sampling(self.alias_edges[(prev, cur)][0], self.alias_edges[(prev, cur)][1])]
                    walk.append(next)
            else:
                break

        return walk

    def get_random_walk(self, chain_length=10, num_iterations=100):
        """
        对整张图生成随机游走链的集合，默认迭代80次，每次对每个节点生成长度为10的链
        返回值是大小为((节点数量x80),10)的巨大list
        80和10都是论文中的实验参数
        """

        print("Start sampling node2vec random walk traces")
        walks = []
        nodes = list(self.graph.nodes())
        for it in range(num_iterations):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.get_one_chain(node, chain_length))
            if it % 10 == 0 and it != 0:
                print("iteration %d ended" % it)

        print("Sampling finished")
        return walks