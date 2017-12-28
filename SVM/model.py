from __future__ import division
import sys
sys.path.append("../")
from constant import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import norm_col_init, weights_init

class SVM(torch.nn.Module):
    def __init__(self, pca_dim = PCA[str(PCA_PERCENTAGE)], classes = CLASSES):
        super(SVM, self).__init__()

        self.fc1 = nn.Linear(pca_dim , classes)

        self.apply(weights_init)
        self.fc1.weight.data = norm_col_init(self.fc1.weight.data, 1.0)
        self.fc1.bias.data.fill_(0)

        self.train()

    def forward(self, inputs):
        # x = F.sigmoid(self.fc1(inputs))
        x = self.fc1(inputs)

        return x

if __name__ == '__main__':
    c = SVM()
    x = c(Variable(torch.ones(453)))
    x = x.max(0)[1].data
    x = x.numpy()[0]
    print(x)
