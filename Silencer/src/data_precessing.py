import pickle
import pandas as pd
import numpy as np
import networkx as nx
import os
from tqdm import tqdm
class Graph():
    def __init__(self, name):
        edgePath = '..//data//' + name + '.csv'
        self.name = name
        self.graph = nx.from_edgelist(pd.read_csv(edgePath).values.tolist())

    def produceNoise(self, ratio, seed=0,switch=False):
        self.A = nx.adjacency_matrix(self.graph).A
        if switch:
            np.random.seed(seed)
        shape = np.shape(self.A)
        randM = np.random.rand(shape[0],shape[1])
        randM = (randM + randM.T)/2
        mask = randM <= ratio
        from scipy import sparse
        if mask.any()==True:
            self.A[mask] = (self.A[mask] + 1) % 2
            self.A = sparse.csr_matrix(self.A)
            resGraph = nx.from_scipy_sparse_matrix(self.A)
            return resGraph
        else:
            return self.graph
    def processingNoise(self, graphNum,):
        if not os.path.exists('..//input//{}//'.format(self.name)):
            os.mkdir('..//input//{}//'.format(self.name))
        noiseList = [0.005,0.01,0.020,0.030,0.040,0.050]
        np.random.seed(0)
        randNoiseSeed = np.random.choice(100,len(noiseList),  replace=False)
        print('randNoiseSeed=', randNoiseSeed)
        for ratioIdx in range(len(noiseList)):
            ratio = noiseList[ratioIdx]
            np.random.seed(randNoiseSeed[ratioIdx])
            randPartSeed = np.random.choice(100,graphNum,  replace=False)
            print('randPartSeed=', randPartSeed)
            for part in tqdm(range(graphNum)):
                savePath = '..//input//{}//{}-ratio_{}-part{}.graph'.format(self.name,self.name, str(ratio), str(part))
                seed = randPartSeed[part]
                graph = self.produceNoise(ratio,seed,True)
                with open(savePath, 'wb') as f:
                    pickle.dump(graph, f)



if __name__ == '__main__':
    name = 'email-Eu-core'
    g = Graph(name)
    g.processingNoise(10)