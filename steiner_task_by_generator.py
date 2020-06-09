# code adapted from https://docs.dgl.ai/tutorials/hetero/1_basics.html?highlight=node%20classification#
# we are doing node classification, which is whether a node in solution or not, later we can do link predicion as well
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
from generate_dataset import generate_steiner_graphs

from sklearn.metrics import confusion_matrix

class Net(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Net, self).__init__()
        self.layers = nn.ModuleList([
            SAGEConv(in_dim, hidden_dim, aggregator_type='lstm'), # need to change if we have more node or edge features 
            SAGEConv(hidden_dim, hidden_dim, aggregator_type='lstm'),
            SAGEConv(hidden_dim, hidden_dim, aggregator_type='lstm')
            ])
        self.linear = nn.Linear(hidden_dim, n_classes)  # predice classes

    def forward(self, g):
        features = th.cat((g.in_degrees().view(-1, 1).float()/g.number_of_nodes(), g.ndata['p']),1)

        for conv in self.layers:
            features = conv(g, features)
        return self.linear(features)

def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label_list).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, th.cat(labels,0).squeeze(1).long()

def main():

    trainset = generate_steiner_graphs(1000, range(20,201,10), save_file=True)
    testset = generate_steiner_graphs(200, range(20,201,10))

    # Use PyTorch's DataLoader and the collate function
    # defined before.
    data_loader = DataLoader(trainset, batch_size=16, shuffle=True,
                             collate_fn=collate)

    # Create model, n_classes is prediction for either in solution or not in sulution
    model = Net(in_dim=2, hidden_dim=64, n_classes=2)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()

    epoch_losses = []
    for epoch in range(200):
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

        if epoch % 5 == 0:
            model.eval()
            # Convert a list of tuples to two lists
            test_X, test_Y = map(list, zip(*testset))
            test_bg = dgl.batch(test_X)
            test_Y = th.cat(test_Y,0).squeeze(1).long()
            test_Y = th.tensor(test_Y).float().view(-1, 1)
            print(test_Y)
            probs_Y = th.softmax(model(test_bg), 1)
            sampled_Y = th.multinomial(probs_Y, 1)
            argmax_Y = th.max(probs_Y, 1)[1].view(-1, 1)
            print('Accuracy of sampled predictions on the test set: {:.4f}%'.format(
                 (test_Y == sampled_Y.float()).sum().item() / len(test_Y) * 100))
            print('Accuracy of argmax predictions on the test set: {:4f}%'.format(
                 (test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100))
            print('Accuracy of all one on the test set: {:4f}%'.format(
                 (test_Y == th.ones(len(test_Y)).float().view(-1, 1)).sum().item() / len(test_Y) * 100))


if __name__ == '__main__':
    main()

