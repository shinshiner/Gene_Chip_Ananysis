from __future__ import print_function, division
import os
import torch
from utils import setup_logger
from model import NET
from torch.autograd import Variable
import torch.nn.functional as F
import time
import numpy as np

dataset_path = "../output/data/dataset_test.npy"
target_path = "../output/data/target_test.npy"

def test(args, shared_model):
    start_time = time.time()
    log = setup_logger(0, 'epoch%d_test' % args.epoch, os.path.join(args.log_dir, 'epoch%d_test_log.txt' % args.epoch))
    log.info('Test time ' + time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)) + ', ' + 'Start testing.')
    local_model = NET()
    local_model.load_state_dict(shared_model.state_dict())
    if args.gpu:
        local_model = local_model.cuda()

    dataset = np.load(dataset_path)
    targets = np.load(target_path)
    correct_cnt = 0

    for idx in range(targets.shape[0]):
        data = dataset[idx]
        data = Variable(torch.from_numpy(data))
        if args.gpu:
            data = data.cuda()

        target = targets[idx]
        output = local_model(data)
        if args.gpu:
            output = output.cpu()
        predict_class = output.max(0)[1].data.numpy()[0]
        if target == predict_class:
            correct_cnt += 1

        if (idx + 1) % 100 == 0:
            log.info('Test time ' + time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)) + ', ' + 'Accuracy: %d / %d\t%0.4f' % (correct_cnt, idx + 1, correct_cnt / (idx + 1)))
    return correct_cnt / targets.shape[0]
