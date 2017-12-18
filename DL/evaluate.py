#coding:utf-8
from __future__ import print_function, division
import os
import re
import numpy as np
import random
from constants import *
from word_embedding import load_word2vec, embedding, embedding_whole
try:
    import xml.etree.cElementTree as ET 
except ImportError:
    import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from model import SA_NET

def tagging(txt):
    print(txt, type(txt))
    for char in txt:
        if not ((char >= 'a' and char <= 'z') or (char >= 'A' and char <= 'Z') or (char >= '0' and char <= '9') or char in " ,.<>/?\\!@#$%^&*()-_+=`~\t;:\'\"[]{}|"):
            return CN
    return EN

def load_model(args, Tag):
    model = SA_NET(Embedding_Dim[Tag])
    saved_state = torch.load(os.path.join(args.model_dir, 'model_%s.dat' % Tag_Name[Tag]))
    model.load_state_dict(saved_state)
    if args.gpu:
        model = model.cuda()
    return model

def evaluate(args, input_file_path, out_file_path):
    xmltree = ET.parse(input_file_path)
    xmlroot = xmltree.getroot()
    language_model = list(map(lambda x: load_word2vec(x), Languages))
    model = list(map(lambda x: load_model(args, x), Languages))

    for review in xmlroot:
        txt = review.text
        if txt[-1] == '\n':
            txt = txt[:-1]
        tag = tagging(txt)
        mat = embedding_whole(language_model[tag], txt, tagging(tag))
        data = Variable(torch.from_numpy(mat))
        if args.gpu:
            data = data.cuda()
        output = model[tag].forward(data)

        if output.data.cpu().numpy()[0] < 0.5:
            review.set("polarity", "-1")
        else:
            review.set("polarity", "1")
    xmltree.write(out_file_path, encoding="utf-8")
