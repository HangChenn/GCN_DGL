import time

import numpy as np
import networkx as nx

import dgl
import dgl.function as fn

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from dgl.nn.pytorch.conv import SAGEConv

from generate_dataset import generate_connected_graphs_G_classfication



class Net(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Net, self).__init__()
        self.layers = nn.ModuleList([
            # please see the model implementation in the following
            # https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/sageconv.html#SAGEConv
            SAGEConv(in_dim, hidden_dim, aggregator_type='mean'), # need to change if we have more node or edge features 
            SAGEConv(hidden_dim, hidden_dim, aggregator_type='mean'),
            SAGEConv(hidden_dim, hidden_dim, aggregator_type='mean')
            
            ])
        self.linear = nn.Linear(hidden_dim, n_classes)  # predice classes

    def forward(self, g):
        features = g.ndata['p']
        # print(g.in_degrees().view(-1, 1).float().size())
        # features = g.in_degrees().view(-1, 1).float()
        for conv in self.layers:
            features = conv(g, features)
        g.ndata['h'] = features
        hg = dgl.mean_nodes(g, 'h')
        return self.linear(hg)

def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label_list).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, th.tensor(labels)


def main():

    trainset = generate_connected_graphs_G_classfication(30*5*5, range(10,201,10))
    testset = generate_connected_graphs_G_classfication(30*5*5, range(10,201,10))
    # Use PyTorch's DataLoader and the collate function
    # defined before.

    # the data_loader batch several graph together to speed up traning procedure
    data_loader = DataLoader(trainset, batch_size=8, shuffle=True,
                             collate_fn=collate)

    # Create model, n_classes is prediction for either is connected or not
    # in_dim is the length of node feature, here we give it random ID
    model = Net(in_dim=1, hidden_dim=16, n_classes=2)


    # difine loss function and optimizer for the whole model, 
    # not just GCN(SAGE) but also the linear layer in the end
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epoch_losses = []
    for epoch in range(301):    
        epoch_loss = 0
        model.train()
        for iter, (bg, label) in enumerate(data_loader):
            prediction = model(bg)

            # back propagate, only use it in training time 
            # loss = loss_func(prediction, label)
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
            # see the performance on testing set
            model.eval()
            # Convert a list of tuples to two lists
            test_X, test_Y = map(list, zip(*testset))
            test_bg = dgl.batch(test_X)
            test_Y = th.tensor(test_Y).float().view(-1, 1)

            # two way to tranform a continue value to binary, usually, max is what we want
            probs_Y = th.softmax(model(test_bg), 1)

            sampled_Y = th.multinomial(probs_Y, 1)
            argmax_Y = th.max(probs_Y, 1)[1].view(-1, 1)

            # test the accuracy.
            print('Accuracy of sampled predictions on the test set: {:.4f}%'.format(
                 (test_Y == sampled_Y.float()).sum().item() / len(test_Y) * 100))
            print('Accuracy of argmax predictions on the test set: {:4f}%'.format(
                 (test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100))
            print('Accuracy of 1 predictions on the test set: {:4f}%'.format(
                 (test_Y == th.ones(len(test_Y)).float().view(-1, 1)).sum().item() / len(test_Y) * 100))



if __name__ == '__main__':
    main()

