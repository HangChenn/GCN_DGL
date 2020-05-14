import torch as th
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import dgl

from os import listdir
from os.path import isfile, join
import math

from input_functions import build_networkx_graph, take_input

class load_steinlib_graphs(object):
    """The dataset class.
    Parameters
    ----------
    num_graphs: int
        Number of graphs in this dataset.
    num_v: list of int
        number of v for graph.
    """
    def __init__(self):
        self.graphs = []

        self._load_Steinlib()

        np.random.shuffle(self.graphs)

        # 80 graphs as train and 20 graphs as test
        num_graphs = math.ceil(0.8*len(self.graphs))
        self.train = self.graphs[:num_graphs]
        self.test = self.graphs[num_graphs:]




    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.train)

    """ return dgl.DGLGraph for given poistion 'idx' """
    def __getitem__(self, idx):
        return self.train[idx]

    def get_testset(self):
        return load_dataset(self.test)

    def _load_Steinlib(self):
        """ 
        Input:  g: networkx graph, 
                node_list: steiner tree terminal, 
                edge_list: steiner tree solution
        -----------------------------------------------------------------------------------
        Output: tuple of items (a,b) -> (DGLGraph(), torch.tensor(m,1))
                a. graph that follow DGL graph with node features(is_terminal) and edge features(weight)
                b. the label list indicate whether the edge in solution
                    (as torch tensor that label 1 to in_solution edge, 0 otherwise)
        """
        def __convert_g(g, node_list, label):
            dgl_g = dgl.DGLGraph()
            dgl_g.from_networkx(g, edge_attrs=['weight'])       
            dgl_g.nodes[node_list].data['p'] = th.ones((len(node_list), 1))
            return dgl_g, label
        
        filename = './I080_graphs/I080/i080-002_1.txt'
        I080_graph_path = './I080_task/I080_task_graph/'
        I080_solution_path = './I080_task/I080_task_solution/'
        filenames = [f for f in listdir(I080_graph_path) if isfile(join(I080_graph_path, f))]

        for filename in filenames:
            filename = filename.rstrip('.txt')
            Stein_graph, Ts = build_networkx_graph(I080_graph_path+filename+'.txt')
            Ts = Ts[0]
            edge_list, _  = take_input(I080_solution_path+filename+'_output.txt')

            n_label = th.zeros([Stein_graph.number_of_nodes(),1])
            for u, v, _ in edge_list:
                n_label[u] = 1
                n_label[v] = 1
            
            self.graphs.append(__convert_g(Stein_graph, Ts, n_label))




    """ 
    Input:  n: number of graph need to generate with ER model
    -----------------------------------------------------------------------------------
    Output: tuple of items (a,b) -> (DGLGraph(), torch.tensor(m,1))
            a. graph that follow DGL graph with node features(is_terminal) and edge features(weight)
            b. the label list indicate whether the edge in solution
                (as torch tensor that label 1 to in_solution edge, 0 otherwise)
    """
    def _gen_ER(self, n):


        for v in self.v_list:
            for _ in range(math.ceil(n/len(self.v_list))):
                while True:
                    g = nx.fast_gnp_random_graph(v, (1+1)*math.log(v)/v)
                    if nx.is_connected(g):
                        break
                copy_g = g.copy()
                node_list = list(g.nodes)
                weight = {}
                for e in g.edges():
                    # random set edge weight from 1 to 10
                    weight[e] = { 'weight' : np.random.randint(low=1, high=10)}
                nx.set_edge_attributes(copy_g, weight)
                # print(g.edges.data())

                # assign terminal randomly to nodes
                np.random.shuffle(node_list)
                first_half_node_l = node_list[:int(0.5*len(node_list))]

                # calculate steiner tree
                ST_f = steiner_tree(copy_g, first_half_node_l)
                
                self.graphs.append(__convert_g(copy_g, first_half_node_l, ST_f.edges()))

class load_dataset(object):
    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graphs)

    """ return dgl.DGLGraph for given poistion 'idx' """
    def __getitem__(self, idx):
        return self.graphs[idx]