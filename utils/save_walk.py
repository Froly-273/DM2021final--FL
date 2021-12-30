def save_walk_result(walk, savepath='../logs/random_walk_trace.txt'):
    """
    保存随机游走的结果
    """
    with open(savepath, 'w') as f:
        for cid, chain in enumerate(walk):
            s = ' '.join(str(node) for node in chain)
            f.writelines([s + '\n'])