import numpy as np
import csv


# ----------------------------------
# 专门用来算余弦相似度，给出预测结果的
# ----------------------------------


def evaluate(fpath='logs/word2vec.txt', testpath='data/lab2_test.csv', output_path='logs/predictions.csv'):
    # 如果想用训练过程中的最好结果做predict，改成fpath='logs/word2vec_best.txt'

    # 读取embedding
    embeddings = {}
    with open(fpath, 'r') as f:
        lines = f.readlines()[1:]  # 不管第一行
        for lid, line in enumerate(lines):
            data = line.strip('\n').split(' ')
            emb = [float(i) for i in data[1:]]
            embeddings[data[0]] = np.array(emb)

    print("Embeddings loaded")

    # 读取测试集
    csvfile = csv.reader(open(testpath, 'r', encoding='utf-8'))
    header = next(csvfile)

    print("Test file loaded")

    # 预测结果用预先相似度衡量
    predictions = {}
    for row in csvfile:
        nid = row[0]
        u = row[1]
        v = row[2]
        if (u not in embeddings.keys()) or (v not in embeddings.keys()):
            predictions[nid] = 0
        else:
            vec_u = embeddings[u]
            vec_v = embeddings[v]
            cos_sim = np.dot(vec_u.reshape(1, -1), vec_v.reshape(-1, 1)) / (np.linalg.norm(vec_u) * np.linalg.norm(vec_v))
            # cos_sim = 1 - distance(vec_u, vec_v)
            # print(float(cos_sim))
            predictions[nid] = cos_sim

    # print(predictions)

    # 写入预测结果
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for key, value in predictions.items():
            writer.writerow([key, float(value)])


if __name__ == 'main':
    evaluate(fpath='../logs/word2vec.txt', testpath='../data/lab2_test.csv', output_path='../logs/predictions.csv')
