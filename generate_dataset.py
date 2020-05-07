import networkx as nx
from networkx.algorithms.approximation import steiner_tree
import dgl
import torch as th
import numpy as np
import math


class generate_connected_graphs_G_classfication(object):
    """The dataset class.
    Parameters
    ----------
    num_graphs: int
        Number of graphs in this dataset.
    num_v: list of int
        number of v for graph.
    """
    def __init__(self, num_graphs, v_list):
        # we generate 5 instance of precoeesed(remove edges) for each random graph

        self.samples_num = 5
        self.num_graphs_per_V = num_graphs/self.samples_num
        self.v_list = v_list

        # graphs is list of tuple with (DGLGraph, label_list)
        self.graphs = []
        # generate graph
        self._generate()
        np.random.shuffle(self.graphs)
        self.graphs = self.graphs[:num_graphs]

        

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graphs)

    """ return dgl.DGLGraph for given poistion 'idx' """
    def __getitem__(self, idx):
        return self.graphs[idx]

    def _generate(self):
        self._gen_ER(self.num_graphs_per_V)

    """ 
    Input:  n: number of graph need to generate with ER model
    -----------------------------------------------------------------------------------
    Store list of tuple [..., (A_i,B_i), ...] -> (DGLGraph(), int)
    -----------------------------------------------------------------------------------        
    Output: None
    """
    def _gen_ER(self, n):
        for v in self.v_list:
            for _ in range(math.ceil(n/len(self.v_list))):
                while True:
                    g = nx.fast_gnp_random_graph(v, (1)*math.log(v)/v)
                    # g = nx.fast_gnp_random_graph(v, 0.5)
                    if nx.is_connected(g):
                        break
                # pick 10% of the edges to remove
                r_e_n = math.ceil(g.number_of_edges()/10)
                def pick_k_edges(g, k):
                    edges = list(g.edges())
                    np.random.shuffle(edges)
                    picked_edges = edges[:k]
                    return picked_edges

                for i in range(self.samples_num):
                    copy_g = g.copy()
                    copy_g.remove_edges_from(pick_k_edges(copy_g, r_e_n))
                    self.graphs.append(self._convert_g(copy_g, 1*nx.is_connected(copy_g)))
    """ 
    Input:  g: networkx graph, 
            label: number to incidate the class of this graph below to
    -----------------------------------------------------------------------------------
    Output: tuple of items (a,b) -> (DGLGraph(), int)
            a. graph that follow DGL graph with node features(is_terminal) and edge features(weight)
            b. the label indicate whether the solution is valid steiner
                
    """
    def _convert_g(self, g, label):
        dgl_g = dgl.DGLGraph()
        dgl_g.from_networkx(g)

        # store random ID as node feature for each node
        random_ID = list(range(len(g.nodes)))
        np.random.shuffle(random_ID)

        ndata = th.zeros((len(random_ID),1))
        for w, i in zip(random_ID, range(len(random_ID))):
            ndata[i] = w
        dgl_g.ndata['p'] = ndata

        return dgl_g, label

'''
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''


class generate_connected_graphs_N_classfication(object):
    """The dataset class.
    Parameters
    ----------
    num_graphs: int
        Number of graphs in this dataset.
    num_v: list of int
        number of v for graph.
    """
    def __init__(self, num_graphs, v_list):
        # we generate 5 instance of precoeesed(remove edges) for each random graph

        self.samples_num = 5
        self.num_graphs_per_V = num_graphs/self.samples_num
        self.v_list = v_list

        # graphs is list of tuple with (DGLGraph, label_list)
        self.graphs = []
        # generate graph
        self._generate()
        np.random.shuffle(self.graphs)
        self.graphs = self.graphs[:num_graphs]

        

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graphs)

    """ return dgl.DGLGraph for given poistion 'idx' """
    def __getitem__(self, idx):
        return self.graphs[idx]

    def _generate(self):
        self._gen_ER(self.num_graphs_per_V)

    """ 
    Input:  n: number of graph need to generate with ER model
    -----------------------------------------------------------------------------------
    Store list of tuple [..., (A_i,B_i), ...] -> (DGLGraph(), int)
    -----------------------------------------------------------------------------------        
    Output: None
    """
    def _gen_ER(self, n):
        for v in self.v_list:
            for _ in range(math.ceil(n/len(self.v_list))):
                while True:
                    g = nx.fast_gnp_random_graph(v, (1)*math.log(v)/v)
                    # g = nx.fast_gnp_random_graph(v, 0.5)
                    if nx.is_connected(g):
                        break
                # pick 10% of the edges to remove
                r_e_n = math.ceil(g.number_of_edges()/10)
                def pick_k_edges(g, k):
                    edges = list(g.edges())
                    np.random.shuffle(edges)
                    picked_edges = edges[:k]
                    return picked_edges

                for i in range(self.samples_num):
                    copy_g = g.copy()
                    copy_g.remove_edges_from(pick_k_edges(copy_g, r_e_n))
                    self.graphs.append(self._convert_g(copy_g))
    """ 
    Input:  g: networkx graph, 
            label: number to incidate the class of this graph below to
    -----------------------------------------------------------------------------------
    Output: tuple of items (a,b) -> (DGLGraph(), int)
            a. graph that follow DGL graph with node features(is_terminal) and edge features(weight)
            b. the label indicate whether the solution is valid steiner
                
    """
    def _convert_g(self, g):
        dgl_g = dgl.DGLGraph()
        dgl_g.from_networkx(g)

        node_ids = list(range(len(g.nodes)))
        np.random.shuffle(node_ids)

        # store random ID for each graph
        ndata = th.zeros((len(node_ids),1))
        for n_id, i in zip(node_ids, range(len(node_ids))):
            ndata[i] = n_id
        dgl_g.ndata['p'] = ndata

        # calculated label for each node
        label = np.asarray(ndata)
        for c in nx.connected_components(g):
            min_id = min(label[np.asarray(list(c))])
            label[np.asarray(list(c))] = min_id
        label = th.tensor(label)

        

        return dgl_g, label


'''
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''



class generate_steiner_graphs(object):
    """The dataset class.
    Parameters
    ----------
    num_graphs: int
        Number of graphs in this dataset.
    num_v: list of int
        number of v for graph.
    """
    def __init__(self, num_graphs, v_list):
        self.node_task = True

        self.num_graphs = num_graphs
        self.v_list = v_list
        # graphs is list of tuple with (DGLGraph, label_list)
        self.graphs = []

        self._generate()
        np.random.shuffle(self.graphs)
        self.graphs = self.graphs[:num_graphs]


    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graphs)

    """ return dgl.DGLGraph for given poistion 'idx' """
    def __getitem__(self, idx):
        return self.graphs[idx]

    def get_testset(self):
        return load_dataset(self.graphs_complement)

    def _generate(self):
        self._gen_ER(self.num_graphs)

    """ 
    Input:  n: number of graph need to generate with ER model
    -----------------------------------------------------------------------------------
    Output: tuple of items (a,b) -> (DGLGraph(), torch.tensor(m,1))
            a. graph that follow DGL graph with node features(is_terminal) and edge features(weight)
            b. the label list indicate whether the edge in solution
                (as torch tensor that label 1 to in_solution edge, 0 otherwise)
    """
    def _gen_ER(self, n):
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
        def __convert_g(g, node_list, edge_list):
            dgl_g = dgl.DGLGraph()
            dgl_g.from_networkx(g, edge_attrs=['weight'])
        
            dgl_g.nodes[node_list].data['p'] = th.ones((len(node_list), 1))

            if self.node_task:
                label = th.zeros([g.number_of_nodes(),1])
                for u, v in edge_list:
                    label[u] = 1
                    label[v] = 1

            # give lable as input feature, this should give 100% accuracy
            # g.ndata['h'] = label
            return dgl_g, label




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
                
                self.graphs.append(__convert_g(copy_g, ST_f.nodes(), ST_f.edges()))
