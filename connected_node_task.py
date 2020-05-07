import time

import numpy as np
import networkx as nx

import dgl
import dgl.function as fn
from dgl import DGLGraph

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from dgl.nn.pytorch.conv import SAGEConv
from generate_dataset import generate_connected_graphs_N_classfication



class Net(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Net, self).__init__()

        self.layers = nn.ModuleList([
            SAGEConv(in_dim, hidden_dim, aggregator_type='mean'),
            # SAGEConv(hidden_dim, hidden_dim, aggregator_type='mean'),
            SAGEConv(hidden_dim, hidden_dim, aggregator_type='mean')
            ])
        self.linear = nn.Linear(hidden_dim, n_classes)  # predice classes

    def forward(self, g):
        # features = th.cat((g.in_degrees().view(-1, 1).float()/g.number_of_nodes(), g.ndata['h']),1)
        features = g.ndata['p']

        for conv in self.layers:
            features = conv(g, features)
        return self.linear(features)

def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label_list_list).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    # we ravel 2D list of label to 1D
    return batched_graph, th.cat(labels,0).squeeze(1).long()


def main():

    max_node_num = 201
    trainset = generate_connected_graphs_N_classfication(30*5*5, range(10,max_node_num,10))
    testset = generate_connected_graphs_N_classfication(30*5*5, range(10,max_node_num,10))

    # Use PyTorch's DataLoader and the collate function
    # defined before.
    data_loader = DataLoader(trainset, batch_size=16, shuffle=True,
                             collate_fn=collate)

    # Create model, n_classes is random node id. noticed, this is only in training
    model = Net(in_dim=1, hidden_dim=32, n_classes=max_node_num)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()

    epoch_losses = []
    for epoch in range(201):
        epoch_loss = 0
        model.train()
        for iter, (bg, label) in enumerate(data_loader):
            prediction = model(bg)
            # train model 
            logp = F.log_softmax(prediction, 1)
            loss = F.nll_loss(logp, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach().item()
        epoch_loss /= (iter + 1)
        print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
        epoch_losses.append(epoch_loss)

        if epoch % 5 == 0:
            model.eval()
            # Convert a list of tuples to two lists
            test_X, test_Y = map(list, zip(*testset))
            test_bg = dgl.batch(test_X)

            probs_Y = th.softmax(model(test_bg), 1)
            argmax_Y = th.max(probs_Y, 1)[1]

            # remember we mash a 2D list into 1D in the training time
            # we need to restore the list back
            new_argmax_Y = []
            for one_graph_label in test_Y:
                new_argmax_Y.append(argmax_Y[:len(one_graph_label)])
                argmax_Y = argmax_Y[len(one_graph_label):]
            argmax_Y = new_argmax_Y

            # convert node label to graph label
            test_Y = th.tensor([1 if len(list(np.unique(la))) > 1 else 0 for la in test_Y])
            argmax_Y = th.tensor([1 if len(list(np.unique(la))) > 1 else 0 for la in argmax_Y])


            print('Accuracy of argmax predictions on the test set: {:4f}%'.format(
                 (test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100))
            print('Accuracy of all one on the test set: {:4f}%'.format(
                 (test_Y == th.ones(len(test_Y)).float()).sum().item() / len(test_Y) * 100))


if __name__ == '__main__':
    main()

