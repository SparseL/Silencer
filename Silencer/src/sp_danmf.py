import numpy as np
from tqdm import tqdm
import networkx as nx
from sklearn.decomposition import NMF
from sklearn import metrics
class DANMF(object):
    """
    Deep autoencoder-like non-negative matrix factorization class.
    """
    def __init__(self, graph, args, label):
        """
        Initializing a DANMF object.
        :param graph: Networkx graph.
        :param args: Arguments object.
        """
        self.graph = graph
        self.A = nx.adjacency_matrix(self.graph)
        self.L = nx.laplacian_matrix(self.graph)
        self.D = self.L+self.A
        self.args = args
        self.p = len(self.args.layers)
        self.bestNMI = { 'NMI':0, 'iteration':0,'membership':[],'ONMI':0}
        self.label = label
    def get_loss_list(self):
        return self.loss
    def setup_z(self, i):
        """
        Setup target matrix for pre-training process.
        """
        if i == 0:
            self.Z = self.A
        else:
            self.Z = self.V_s[i-1]
    def sklearn_pretrain(self, i,seed=None):
        """
        Pretraining a single layer of the model with sklearn.
        :param i: Layer index.
        """
        if seed!= None:
            nmf_model = NMF(n_components=self.args.layers[i],
                            init="random",
                            random_state=seed,
                            max_iter=self.args.pre_iterations,) #random_state=self.args.seed,

            U = nmf_model.fit_transform(self.Z)
            V = nmf_model.components_
            return U, V
        if seed== None:
            nmf_model = NMF(n_components=self.args.layers[i],
                            init="random",
                            max_iter=self.args.pre_iterations) #random_state=self.args.seed,
            U = nmf_model.fit_transform(self.Z)
            V = nmf_model.components_
            return U, V
    def pre_training(self,seed=None):
        """
        Pre-training each NMF layer.
        """
        print("\nLayer pre-training started. \n")
        self.U_s = []
        self.V_s = []
        for i in range(self.p):
            self.setup_z(i)
            U, V = self.sklearn_pretrain(i,seed)
            self.U_s.append(U)
            self.V_s.append(V)
    def calculate_cost(self):
        """
        Calculate loss.
        :param i: Global iteration.
        """
        reconstruction_loss_1 = np.linalg.norm(self.A-self.P.dot(self.V_s[-1]), ord="fro")**2
        reconstruction_loss_2 = np.linalg.norm(self.V_s[-1]-self.A.dot(self.P).T, ord="fro")**2
        regularization_loss = np.trace(self.V_s[-1].dot(self.L.dot(self.V_s[-1].T)))
        loss = reconstruction_loss_1+reconstruction_loss_2+self.args.lamb*regularization_loss
        self.loss.append(loss)
    def setup_Q(self):
        """
        Setting up Q matrices.
        """
        self.Q_s = [None for _ in range(self.p+1)]
        self.Q_s[self.p] = np.eye(self.args.layers[self.p-1])
        for i in range(self.p-1, -1, -1):
            self.Q_s[i] = np.dot(self.U_s[i], self.Q_s[i+1])
    def update_U(self, i):
        """
        Updating left hand factors.
        :param i: Layer index.
        """
        if i == 0:
            R = self.U_s[0].dot(self.Q_s[1].dot(self.VpVpT).dot(self.Q_s[1].T))
            R = R+self.A_sq.dot(self.U_s[0].dot(self.Q_s[1].dot(self.Q_s[1].T)))
            Ru = 2*self.A.dot(self.V_s[self.p-1].T.dot(self.Q_s[1].T))
            self.U_s[0] = (self.U_s[0]*Ru)/np.maximum(R, 10**-10)
        else:
            R = self.P.T.dot(self.P).dot(self.U_s[i]).dot(self.Q_s[i+1]).dot(self.VpVpT).dot(self.Q_s[i+1].T)
            R = R+self.A_sq.dot(self.P).T.dot(self.P).dot(self.U_s[i]).dot(self.Q_s[i+1]).dot(self.Q_s[i+1].T)
            Ru = 2*self.A.dot(self.P).T.dot(self.V_s[self.p-1].T).dot(self.Q_s[i+1].T)
            self.U_s[i] = (self.U_s[i]*Ru)/np.maximum(R, 10**-10)
    def update_P(self, i):
        """
        Setting up P matrices.
        :param i: Layer index.
        """
        if i == 0:
            self.P = self.U_s[0]
        else:
            self.P = self.P.dot(self.U_s[i])
    def update_V(self, i):
        """
        Updating right hand factors.
        :param i: Layer index.
        """
        if i < self.p-1:
            Vu = 2*self.A.dot(self.P).T
            Vd = self.P.T.dot(self.P).dot(self.V_s[i])+self.V_s[i]
            self.V_s[i] = self.V_s[i] * Vu/np.maximum(Vd, 10**-10)
        else:
            Vu = 2*self.A.dot(self.P).T+(self.args.lamb*self.A.dot(self.V_s[i].T)).T
            Vd = self.P.T.dot(self.P).dot(self.V_s[i])
            Vd = Vd + self.V_s[i]+(self.args.lamb*self.D.dot(self.V_s[i].T)).T
            self.V_s[i] = self.V_s[i] * Vu/np.maximum(Vd, 10**-10)
    def training(self):
        """
        Training process after pre-training.
        """
        print("\n\nTraining started. \n")
        self.loss = []
        self.A_sq = self.A.dot(self.A.T)
        for iteration in tqdm(range(self.args.iterations)):
            self.setup_Q()
            self.VpVpT = self.V_s[self.p-1].dot(self.V_s[self.p-1].T)
            for i in range(self.p):
                self.update_U(i)
                self.update_P(i)
                self.update_V(i)
            if self.args.calculate_loss:
                self.calculate_cost()
        return self.get_membership()
    def get_membership(self):
        self.membership = np.argmax(self.P, axis=1)
        return self.membership
    def calNMI(self,):
        res = np.argmax(self.P, axis=1)
        NMI = metrics.normalized_mutual_info_score(res, self.label)
        return NMI
class spDANMF(DANMF):
    """
    Slef-paced DANMF.
    """
    def __init__(self, graph, args, label):
        super(spDANMF, self).__init__(graph, args, label)
        self.number_of_nodes = len(self.graph.nodes())
        self.number_of_coummunity = self.args.layers[-1]
        self.W = np.ones((self.number_of_coummunity, self.number_of_nodes))     # (community_number, node_number)
        self.Li = np.ones((self.number_of_coummunity, self.number_of_nodes))     # (community_number, node_number)
        self.gama = 0
        self.eta = args.eta
    def Sp_update_U(self, i):
        if i == 0:
            Pi = ((self.U_s[i]@ self.Q_s[i+1] @ self.V_s[-1])) @ (self.V_s[-1].T) @ (self.Q_s[i+1].T)
            Pi += self.A @ ((self.W)* (self.Q_s[i+1].T @ self.U_s[i].T @ self.A)).T @ (self.Q_s[i+1].T)
            up = ((self.A.A)@self.V_s[-1].T + self.A @ ((self.W)*self.V_s[-1]).T) @ self.Q_s[i+1].T
            self.U_s[i] = (self.U_s[i]*up)/np.maximum(Pi, 10**-10)
        else:
            Pi = self.P.T @ ((self.P @ self.U_s[i]@ self.Q_s[i+1] @ self.V_s[-1])) @ (self.V_s[-1].T) @ (self.Q_s[i+1].T)
            Pi += self.P.T @ self.A @ (self.W* (self.Q_s[i+1].T @ self.U_s[i].T @ self.P.T @ self.A)).T @ (self.Q_s[i+1].T)
            up = self.P.T @ ((self.A.A)@self.V_s[-1].T + self.A @ (self.W*self.V_s[-1]).T) @ self.Q_s[i+1].T
            self.U_s[i] = (self.U_s[i]*up)/np.maximum(Pi, 10**-10)
    def Sp_update_V(self, i):
        """
        Updating right hand factors.
        :param i: Layer index.
        """
        if i < self.p-1:
            pass
        else:
            Vu = self.P.T@(self.A.A) + (self.W*(self.P.T@self.A)) + self.args.lamb*self.V_s[i]@self.A.T
            Vd = self.P.T@((self.P@self.V_s[i])) + (self.W*self.V_s[i]) + self.args.lamb*self.V_s[i]@self.D.T
            self.V_s[i] = self.V_s[i] * Vu/np.maximum(Vd, 10**-10)
    def cal_li(self):
        li = np.array((self.V_s[-1] - self.Q_s[0].T @ self.A))
        Li = li*li
        return np.array(Li)
    def newCalWeight(self,Li):
        if self.gama == 0:
            self.gama = pow(Li.mean(), 0.5)
        def conditionMatrixAssignment(M, gama):
            Matrix = M.copy()
            condition1 = pow(gama/(gama+1),2)
            condition2 = gama*gama
            mask1 = Matrix<=condition1
            mask2 = Matrix>=condition2
            Matrix[mask1] = 1
            Matrix[mask2] = 0
            mask1 += mask2
            Matrix[~mask1] = 1.0/pow(Matrix[~mask1],0.5) - 1.0/gama
            return np.array(Matrix)
        self.W = conditionMatrixAssignment(Li, self.gama)
    def calculate_cost(self):
        reconstruction_loss_1 = np.linalg.norm(self.A-self.P.dot(self.V_s[-1]), ord="fro")**2
        reconstruction_loss_2 = np.linalg.norm(self.W*(self.V_s[-1]-self.A.dot(self.P).T), ord="fro")**2
        loss = reconstruction_loss_1+reconstruction_loss_2
        self.loss.append(loss)
    def training(self):
        self.loss = []
        self.A_sq = self.A.dot(self.A.T)
        for bigIteration in tqdm(range(self.args.bigIterations), desc="Self-paced learning: "):
            self.setup_Q()
            Li = self.cal_li()
            self.newCalWeight(Li)
            for iteration in range(self.args.iterations):
                self.setup_Q()
                self.VpVpT = self.V_s[self.p-1].dot(self.V_s[self.p-1].T)
                for i in range(self.p):
                    self.Sp_update_U(i)
                    self.update_P(i)
                    self.Sp_update_V(i)
                if self.args.calculate_loss:
                    self.calculate_cost()
        return self.get_membership()



