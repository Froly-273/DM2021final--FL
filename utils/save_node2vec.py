def save_node2vec_result(id2word, embedding, N, output_dir='../logs/embedding_output.txt'):
    """
    保存word2vec的结果,其大小为"词数"x"embedding size(100)"
    也就是说每一个单词(在本例中是节点序号)都被encode成了一个100维的向量
    """
    fout = open(output_dir, 'w')
    fout.write('Active nodes: %d, dimension of hidden layer: %d\n' % (len(id2word), N))
    for wid, w in id2word.items():
        e = embedding[wid]
        e = ' '.join(map(lambda x: str(x), e))
        fout.write('%s %s\n' % (w, e))
    fout.close()
    print("word2vec result saved.")