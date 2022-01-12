from model.randomwalk import RandomWalk
import pandas as pd
import networkx as nx
from utils.save_walk import save_walk_result


def generate_random_walk(fpath='../data/lab2_edge.csv', walkpath='../logs/random_walk_trace.txt', p=1.0, q=1.0, num_iterations=100):
    dataframe = pd.read_csv(fpath)
    graph = nx.from_pandas_edgelist(dataframe, 'source', 'target', create_using=nx.Graph())
    graph = graph.to_directed()

    rw = RandomWalk(graph, p, q)
    rw.preprocess()                 # 获得alias_nodes和alias_edges
    walks = rw.get_random_walk(num_iterations=num_iterations)    # 获得一组随机游走链
    save_walk_result(walks, walkpath)         # 以.txt格式保存这些链

if __name__ == "__main__":
    for i in range(10):
        root_dir = '../data/noniid/'
        generate_random_walk(root_dir+'client'+str(i)+'.csv', root_dir+'client'+str(i)+'_walk.txt')
        print(i)