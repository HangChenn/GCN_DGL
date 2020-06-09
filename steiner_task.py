# code adapted from https://docs.dgl.ai/tutorials/hetero/1_basics.html?highlight=node%20classification#
# we are doing node classification, which is whether a node in solution or not, later we can do link predicion as well
import time

import numpy as np
import networkx as nx
from networkx.algorithms.approximation import steiner_tree

import dgl
import dgl.function as fn
from dgl import DGLGraph

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from dgl.nn.pytorch.conv import SAGEConv, GatedGraphConv
from load_dataset import load_steinlib_graphs

from itertools import combinations
from statistics import mean 

from sklearn.metrics import confusion_matrix

class Net(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        # noted, lstm and pool have the best performnace reported in GraphSage paper, the typical number of hidden layers is 2
        super(Net, self).__init__()
        self.layers = nn.ModuleList([
            SAGEConv(in_dim, hidden_dim, aggregator_type='lstm', feat_drop=0.5, activation=F.relu), # need to change if we have more node or edge features 
            SAGEConv(hidden_dim, hidden_dim, aggregator_type='lstm', feat_drop=0.5,activation=F.relu),
            # SAGEConv(hidden_dim, hidden_dim, aggregator_type='mean'),
            SAGEConv(hidden_dim, n_classes, aggregator_type='lstm', activation=None)
            # SAGEConv(in_dim, hidden_dim, aggregator_type='pool', feat_drop=0.5, activation=F.relu), # need to change if we have more node or edge features 
            # SAGEConv(hidden_dim, hidden_dim, aggregator_type='pool', feat_drop=0.5,activation=F.relu),
            # # SAGEConv(hidden_dim, hidden_dim, aggregator_type='mean'),
            # SAGEConv(hidden_dim, n_classes, aggregator_type='pool', activation=None)
            ])
        # self.linear = nn.Linear(hidden_dim, n_classes)  # predice classes

    def forward(self, g):
        features = th.cat((g.in_degrees().view(-1, 1).float()/g.number_of_nodes(), g.ndata['p']),1)

        for conv in self.layers:
            features = conv(g, features)
        # return self.linear(features)
        return features

def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label_list).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, th.cat(labels,0).squeeze(1).long()

def main():
    # t_task = 'I080'
    # t_task = 'I080_subdivide'
    # t_task = 'I080_node_weight'
    t_task = "ER_4logn"
    trainset = load_steinlib_graphs(task=t_task)
    
    testset = trainset.get_testset()

    # Use PyTorch's DataLoader and the collate function
    # defined before.
    data_loader = DataLoader(trainset, batch_size=16, shuffle=True,
                             collate_fn=collate)

    # Create model, n_classes is prediction for either in solution or not in sulution
    if t_task == 'I080_node_weight':
        model = Net(in_dim=3, hidden_dim=64, n_classes=2)
    else:
        model = Net(in_dim=2, hidden_dim=64, n_classes=2)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    model.train()

    epoch_losses = []
    for epoch in range(1,201):
        epoch_loss = 0
        model.train()
        for iter, (bg, label) in enumerate(data_loader):

            prediction = model(bg)

            logp = F.log_softmax(prediction, 1)
            loss = F.nll_loss(logp, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach().item()
        epoch_loss /= (iter + 1)
        print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
        epoch_losses.append(epoch_loss)

        if epoch % 10 == 0:
            model.eval()
            # Convert a list of tuples to two lists
            test_X, test_Y = map(list, zip(*testset))
            test_bg = dgl.batch(test_X)
            test_Y = th.cat(test_Y,0).squeeze(1).long()
            test_Y = th.tensor(test_Y).float().view(-1, 1)
            probs_Y = th.softmax(model(test_bg), 1)
            sampled_Y = th.multinomial(probs_Y, 1)
            argmax_Y = th.max(probs_Y, 1)[1].view(-1, 1)
            print(confusion_matrix(test_Y, argmax_Y))
            print('Accuracy of sampled predictions on the test set: {:.4f}%'.format(
                 (test_Y == sampled_Y.float()).sum().item() / len(test_Y) * 100))
            print('Accuracy of argmax predictions on the test set: {:4f}%'.format(
                 (test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100))
            print('Accuracy of all one on the test set: {:4f}%'.format(
                 (test_Y == th.ones(len(test_Y)).float().view(-1, 1)).sum().item() / len(test_Y) * 100))
            connected_num = 0
            all_num = 0
            no_output = 0
            weight_percent = []
            weight_percent_approx= []
            for g, node_labels in testset:
                
                probs_Y = th.softmax(model(g), 1)
                argmax_Y = th.max(probs_Y, 1)[1]

                # print("solution: ", end='')
                real_weight, _ = calculated_weight(g, node_labels.view(1,-1)[0])
                # print("prediction: ", end='')
                predicited_weight, is_connected = calculated_weight(g, argmax_Y)
                two_approx_weight = calculated_two_approx(g)
                if is_connected == 1 and False:
                    print("~~~~ this graph is connected")
                    print("weight is: {:2f}".format(predicited_weight/real_weight))
                weight_percent.append(predicited_weight/real_weight)
                weight_percent_approx.append(two_approx_weight/real_weight)


                connected_num += is_connected
                all_num += 1

            print("the connected percentage is {:2f}%".format(connected_num/all_num*100))
            print(weight_percent)
            print("average ratio: {:2f}, max ratio {:2f}".format(mean(weight_percent), max(weight_percent)))
            print(weight_percent_approx)
            print("two approx ratio: {:2f}, max ratio: {:2f}".format(mean(weight_percent_approx), max(weight_percent_approx)))


def calculated_two_approx(graph):
    terminal = np.where(graph.ndata['t'].view(1,-1)[0] == 1)[0]
    graph = graph.to_networkx(edge_attrs=['weight']).to_undirected()
    T = steiner_tree(graph, terminal)
    T_weight = 0
    for _, _, edge_weight in T.edges.data('weight', default=1):
        T_weight += int(edge_weight)
    return T_weight

def calculated_weight(graph, node_list):
    terminal = np.where(graph.ndata['t'].view(1,-1)[0] == 1)[0]
    graph = graph.to_networkx(edge_attrs=['weight']).to_undirected()

    # print(node_list)
    node_list = np.where(node_list == 1)[0]
    # print(node_list)
    sub_g = graph.subgraph(node_list)



    # add terminal to the solution 
    final_node_list = list(set(node_list) | set(terminal))

    gcn_solution = graph.subgraph(final_node_list)
    if not nx.is_connected(gcn_solution):
        new_weight = {}
        for u,v, fea_dict in graph.edges(data=True):
            new_weight[(u,v)] = { 'weight_mult_msts' : fea_dict['weight']}

        gcn_f = nx.minimum_spanning_tree(gcn_solution)

        edge_weight_sum = 0
        components_node = []

        # set edge weight inside msts to 0
        for c in nx.connected_components(gcn_f):
            for u, v, fea_dict in gcn_f.subgraph(c).edges(data=True):
                edge_weight_sum += fea_dict['weight']
                new_weight[(u,v)] = { 'weight_mult_msts' : 0}
            components_node.append(c.pop())
        nx.set_edge_attributes(graph, new_weight)

        # print(edge_weight_sum)

        # 
        for i in range(len(components_node)-1):
            u = components_node[i]
            v = components_node[i+1]
            path_data = nx.shortest_path(graph, source=u, target=v, weight='weight_mult_msts')
            # print(path_data)
            for j  in range(len(path_data)-1):
                u_p = path_data[j]
                v_p = path_data[j+1]
                
                edge_weight_sum += graph[u_p][v_p]['weight_mult_msts']

        T_weight = int(edge_weight_sum)
    else:
        T_g = graph.subgraph(final_node_list)
        T = nx.minimum_spanning_tree(T_g, weight='weight')
        T_weight = 0
        for _, _, edge_weight in T.edges.data('weight', default=1):
            T_weight += int(edge_weight)
        # print(T_weight)

    return T_weight, 0 if len(node_list) == 0 else nx.is_connected(sub_g)*1

# def calculated_weight(graph, node_list):
#     terminal = np.where(graph.ndata['t'].view(1,-1)[0] == 1)[0]
#     graph = graph.to_networkx(edge_attrs=['weight']).to_undirected()
    
#     # print(node_list)
#     node_list = np.where(node_list == 1)[0]
#     # print(node_list)
#     sub_g = graph.subgraph(node_list)



#     T_weight = -1
#     # add terminal when finding 2-aprox Steiner tree
#     final_node_list = list(set(node_list) | set(terminal))
#     if not nx.is_connected(graph.subgraph(final_node_list)):
#         final_node_list = set()
#         for u,v in list(combinations(node_list, 2)):
#             final_node_list |= set(nx.shortest_path(graph, source=u, target=v,weight='weight'))
#             # for path in nx.all_shortest_paths(graph, source=u, target=v, weight='weight'):
#             #     final_node_list |= set(path)
#         final_node_list = list(final_node_list)
#     # print("Final node: ", final_node_list,sep='')

#     T_g = graph.subgraph(final_node_list)
#     T = nx.minimum_spanning_tree(T_g)
#     T_weight = 0
#     for _, _, edge_weight in T.edges.data('weight', default=1):
#         T_weight += int(edge_weight)
#         # print(T_weight)

#     return T_weight, 0 if len(node_list) == 0 else nx.is_connected(sub_g)*1


if __name__ == '__main__':
    main()

