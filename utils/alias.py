import numpy as np


def alias_setup(probs):
    """
    这是一个utils函数，
    Alias抽样算法，这里完成的是第一步：把一个概率分布拉成1xN的矩形，N为probs的长度
    之所以叫alias_setup是因为他没有做sampling

    https://blog.csdn.net/guaidoukx/article/details/87949095

    :param probs: 一个表示概率的numpy数组

    :returns: J，表示每个柱子被填补的部分来自于哪个数组；q，表示每个柱子本身的概率
    """

    N = len(probs)
    q = np.zeros(N)
    J = np.zeros(N, dtype=np.int)

    smaller = []
    larger = []
    for idx, prob in enumerate(probs):
        # 首先给每个柱子都乘以N（N为probs数组的长度）
        q[idx] = N * prob
        if q[idx] < 1.0:
            smaller.append(idx)
        else:
            larger.append(idx)

    while len(smaller) > 0 and len(larger) > 0:
        # smaller和larger数组用来记录每个柱子的高度是否比1小（大）
        # 算法迭代至所有柱子的高度都为1停止
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        # 高于1的柱子抽掉一部分去填补低于1的柱子
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_sampling(J, q):
    """
    接上面的步骤，根据J（每个柱子小于1的部分来自哪个其他柱子）和q（每个柱子原始概率值）使用Alias sampling方法采样出一个概率值
    """

    N = len(J)

    idx = int(np.floor(np.random.rand() * N))
    if np.random.rand() < q[idx]:
        return idx
    else:
        return J[idx]