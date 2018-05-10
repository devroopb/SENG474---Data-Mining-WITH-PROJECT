#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


NUM_BATCH = 500
BATCH_SIZE = 256
PRINT_INTERVAL = 20

# A simple framework to work with pytorch
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # This architecture will not work. 
        # Explain this is true for part c, and
        # design something that will work for part a
        ##############################
        #
        #  put your code for the architecture here
        #
        ##############################
        self.l1 = nn.Linear(2,1)

    def forward(self, x):
        ##############################
        #
        #  put your code for the forward pass here
        #
        ##############################
        return F.sigmoid(self.l1(x))

# may be of use to you
# returns the percentage of predictions (greater than threshold) 
# that are equal to the labels provided
def percentage_correct(pred, labels, threshold = 0.5):
    ##############################
    #
    #  put your code to calculate % correct here
    #
    ##############################
    return 0

# This code generates 2D data with label 1 if the point lies
# outside the unit circle.
def get_batch(batch_size):
    # Data has two dimensions, they are randomly generated
    data = (torch.rand(batch_size,2)-0.5)*2.5
    # square them and sum them to define the decision boundary
    # (x_1)^2 + (x_2)^2 = 1
    square = torch.mul(data,data)
    square_sum = torch.sum(square,1,keepdim=True)
    # Generate the labels
    # outside the circle is 1
    labels = square_sum>1
    return Variable(data), Variable(labels.float())

def plot_decision_boundary(data_in, preds):
    dic= defaultdict(lambda: "r")
    dic[0] = 'b'
    colour = list(map(lambda x: dic[x[0]], preds.data.numpy()>0.5))
    x = data_in.data.numpy()[:,0]
    y = data_in.data.numpy()[:,1]
    plt.clf()
    plt.scatter(x,y,c=colour)
    plt.title("Decision Boundary of a Neural Net Trained to Classify the Unit Circle")
    plt.show()
    # May be of use for saving your plot:    plt.savefig(filename)

# Here's the spot where you'll do your batches of optimization
model = Classifier()
o = torch.optim.SGD(model.parameters(), lr = 0.001)
loss = nn.BCELoss()
for i in range(NUM_BATCH):
    data, labels = get_batch(BATCH_SIZE)
    ##############################
    #
    #  put your code for calling updates here
    #
    ##############################

# plot decision boundary for new data
d, labels = get_batch(BATCH_SIZE)
plot_decision_boundary(d, model(d))



