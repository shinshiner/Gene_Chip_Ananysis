from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import norm_col_init, weights_init

class NET(torch.nn.Module):
    def __init__(self, pca_dim = 453, classes = 76):
        super(NET, self).__init__()

        self.fc1 = nn.Linear(pca_dim , 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, classes)

        self.apply(weights_init)

        self.fc1.weight.data = norm_col_init(self.fc1.weight.data, 1.0)
        self.fc1.bias.data.fill_(0)
        self.fc2.weight.data = norm_col_init(self.fc2.weight.data, 1.0)
        self.fc2.bias.data.fill_(0)
        self.fc3.weight.data = norm_col_init(self.fc3.weight.data, 1.0)
        self.fc3.bias.data.fill_(0)
        self.fc4.weight.data = norm_col_init(self.fc4.weight.data, 1.0)
        self.fc4.bias.data.fill_(0)
        self.fc5.weight.data = norm_col_init(self.fc5.weight.data, 1.0)
        self.fc5.bias.data.fill_(0)

        self.train()

    def forward(self, inputs):
        x = F.relu(F.dropout(self.fc1(inputs)))
        x = F.relu(F.dropout(self.fc2(x)))
        x = F.relu(F.dropout(self.fc3(x)))
        x = F.relu(F.dropout(self.fc4(x)))
        x = self.fc5(x)

        return x

if __name__ == '__main__':
    c = NET()
    x = c(Variable(torch.ones(453)))
    x = x.max(0)[1].data
    x = x.numpy()[0]
    print(x)
