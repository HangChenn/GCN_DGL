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
    def __init__(self, task='I080_node_weight'):
        self.graphs = []

        self._load_Steinlib(task)

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

    def _load_Steinlib(self, task):
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
            if bool(nx.get_node_attributes(g, 'node_weight')):
                dgl_g.from_networkx(g, node_attrs=['node_weight'], edge_attrs=['weight']) 
            else:
                dgl_g.from_networkx(g, edge_attrs=['weight'])       
            dgl_g.nodes[node_list].data['t'] = th.ones((len(node_list), 1))
            dgl_g.ndata['p'] = dgl_g.ndata['t']
            if bool(nx.get_node_attributes(g, 'node_weight')):
                dgl_g.ndata['p'] = th.cat(  (dgl_g.ndata['p'],
                                            dgl_g.ndata['node_weight'].view(-1,1).float())
                                            ,1)

            return dgl_g, label
        
        if task == 'I080':
            graph_path = './I080_task/I080_task_graph/'
            solution_path = './I080_task/I080_task_solution/'
        if task == 'I080_subdivide':
            graph_path = './I080_task/subdivide/graph/'
            solution_path = './I080_task/subdivide/solution/'
        if task == 'I080_node_weight':
            graph_path = './I080_task/node_weight/graph/'
            solution_path = './I080_task/node_weight/solution/'
        if task == 'ER_4logn':
            graph_path = './ER_4logn_task/graph/'
            solution_path = './ER_4logn_task/solution/'

        filenames = [f.rstrip('.txt') for f in listdir(graph_path) if isfile(join(graph_path, f))]
        if task == 'ER_4logn':
            filenames = [f.rstrip('_output.txt') for f in listdir(solution_path) if isfile(join(solution_path, f))]
        print(filenames[0])
        for filename in filenames:
            Stein_graph, Ts = build_networkx_graph(graph_path+filename+'.txt')
            Ts = Ts[0]
            if task == 'I080' or task == 'ER_4logn':
                edge_list, _, _  = take_input(solution_path+filename+'_output.txt')
            if task == 'I080_subdivide':
                edge_list, _, _  = take_input(solution_path+filename.rstrip('_subdivided')+'_output_subdivided.txt')
            if task == 'I080_node_weight':
                edge_list, _, _  = take_input(solution_path+filename.rstrip('_node_weighted')+'_output_node_weighted.txt')

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