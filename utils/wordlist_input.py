import numpy as np
from collections import deque

# ---------------------------------
# word2vec(skipgram)的预处理
# 输入一个已经是word id的句子（在本例中是随机游走生成的chain），预处理得到其中的词语对
# ---------------------------------


class InputData:
    """
    储存输入数据的类，输入的文件名file_name为保存random wolk结果的文件
    """

    def __init__(self, file_name, min_count=10, client_idx=0, num_clients=0, iid=True, length_seed=None, batch_size=10000):
        self.input_file_name = file_name
        self.client_idx = client_idx
        self.num_clients = num_clients
        self.iid = iid
        self.length_seed = length_seed
        self.batch_size = batch_size
        self.get_words(min_count)
        self.word_pair_catch = deque()
        self.init_sample_table()

    def get_words(self, min_count):
        self.input_file = open(self.input_file_name, encoding='utf-8')
        self.sentence_length = 0
        self.sentence_count = 0
        word_frequency = dict()

        # 常规的，非分布式学习
        if not self.num_clients:
            for line in self.input_file:
                self.sentence_count += 1
                line = line.strip().split(' ')
                self.sentence_length += len(line)
                for w in line:
                    try:
                        word_frequency[w] += 1
                    except:
                        word_frequency[w] = 1
        # 分布式学习
        else:
            assert self.client_idx < self.num_clients
            lines = self.input_file.readlines()
            self.sentence_count = len(lines)

            # 确定这个client获取的数据范围在哪里
            # 数据iid情况
            if self.iid:
                start_line = int(self.client_idx * self.sentence_count / self.num_clients)
                end_line = start_line + int(self.sentence_count / self.num_clients) - 1
                lines = lines[start_line : end_line]
            # 数据非iid情况
            else:
                # 用户不指定随机长度（0~1之间）那就随机生成
                if self.length_seed is None:
                    self.length_seed = np.random.rand()
                start_line = int(self.client_idx * self.sentence_count / self.num_clients)
                end_line = start_line + max(int(self.length_seed * self.sentence_count / self.num_clients) - 1, self.batch_size)
                lines = lines[start_line: end_line]

            self.sentence_count = len(lines)
            print("client %d: got lines [%d, %d] in the dataset" % (self.client_idx, start_line, end_line))

            for line in lines:
                line = line.strip().split(' ')
                self.sentence_length += len(line)
                for w in line:
                    try:
                        word_frequency[w] += 1
                    except:
                        word_frequency[w] = 1

        self.word2id = dict()
        self.id2word = dict()
        wid = 0
        self.word_frequency = dict()
        for w, c in word_frequency.items():
            if c < min_count:
                self.sentence_length -= c
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1
        self.word_count = len(self.word2id)

    def init_sample_table(self):
        self.sample_table = []
        sample_table_size = 1e8
        pow_frequency = np.array(list(self.word_frequency.values()))**0.75
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * sample_table_size)
        for wid, c in enumerate(count):
            self.sample_table += [wid] * int(c)
        self.sample_table = np.array(self.sample_table)

    def get_batch_pairs(self, batch_size, window_size):
        while len(self.word_pair_catch) < batch_size:
            sentence = self.input_file.readline()
            if sentence is None or sentence == '':
                self.input_file = open(self.input_file_name, encoding='utf-8')
                sentence = self.input_file.readline()
            word_ids = []
            for word in sentence.strip().split(' '):
                try:
                    word_ids.append(self.word2id[word])
                except:
                    continue

            # print(word_ids)     # [0, 1, 2, 3, 4, 5, 5, 6, 7]

            for i, u in enumerate(word_ids):
                # 那么这个window size控制的是这个词前后各window_size-1个元素，比如4就是(4,0),(4,1),(4,2),...,(4,7)
                for j, v in enumerate(
                        word_ids[max(i - window_size, 0):i + window_size]):
                    assert u < self.word_count
                    assert v < self.word_count
                    if i == j:
                        continue
                    self.word_pair_catch.append((u, v))

        # print(self.word_pair_catch) # deque([(0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 2), (1, 3), (1, 4),...

        batch_pairs = []
        for _ in range(batch_size):
            batch_pairs.append(self.word_pair_catch.popleft())

        # print(batch_pairs)  # [(0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 2), (1, 3), (1, 4),...跟上面那个一样，截取了前batch_size个元素

        return batch_pairs

    # @profile
    def get_neg_v_neg_sampling(self, pos_word_pair, count):
        """
        负采样，也就是拿上下文跟随便一个不是中间那个词的搭配做采样
        """
        neg_v = np.random.choice(
            self.sample_table, size=(len(pos_word_pair), count)).tolist()
        return neg_v

    def evaluate_pair_count(self, window_size):
        return self.sentence_length * (2 * window_size - 1) - (
            self.sentence_count - 1) * (1 + window_size) * window_size


if __name__ == "__main__":
    fname = '../logs/random_walk_trace.txt'
    num_clients = 10
    client_idx = 0
    data = InputData(fname, client_idx=client_idx, num_clients=num_clients)

    # 记录一下输出
    # Client 0 got lines [0, 167139] in the dataset
    # Client 1 got lines [167140, 334279] in the dataset
    # Client 9 got lines [1504260, 1671399] in the dataset
    # 说明数据集的划分很正常
