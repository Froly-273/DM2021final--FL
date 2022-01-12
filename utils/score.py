import csv

def AUCscore(pred_path='logs/baseline/predictions.csv', gt_path='data/lab2_truth.csv'):
    # 读predictions
    predictions = []
    with open(pred_path, 'r', newline='') as f:
        r1 = csv.reader(f)
        for row in r1:
            predictions.append(float(row[1]))

    # 读ground truth
    # 按照AUC的指标，分别取出ground truth中值为0和1的子集的坐标
    # if = inverted file，if0和if1分别索引ground truth中值为0和1的点
    gt = []
    if0 = []
    if1 = []
    with open(gt_path, 'r', newline='') as f:
        r = csv.reader(f)
        idx = 0
        for row in r:
            if row[0] == 'id':  # 跳过第一行
                continue
            gt.append(float(row[1]))
            if row[1] == '1':
                if1.append(idx)
            else:
                if0.append(idx)
            idx += 1

    assert len(predictions) == len(gt)
    len1 = len(if1)
    len0 = len(if0)
    correct = 0

    # AUC计数
    for i in range(len0):
        for j in range(len1):
            correct += (predictions[if0[i]] < predictions[if1[j]])

    score = correct / (len1 * len0)
    return score