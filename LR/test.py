from __future__ import print_function, division
import torch
from model import LR_NET
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import f1_score
import torch.nn.functional as F
import time

def test(args, shared_model, dataset, targets, log):
    start_time = time.time()
    log.info('Test time ' + time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)) + ', ' + 'Start testing.')
    local_model = LR_NET()
    local_model.load_state_dict(shared_model.state_dict())
    if args.gpu:
        local_model = local_model.cuda()

    correct_cnt = 0
    predictions = np.zeros([targets.shape[0]], dtype=np.int64)

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
        predictions[idx] = predict_class
        if target == predict_class:
            correct_cnt += 1
        # else:
        #     print(predict_class)

        # if (idx + 1) % 100 == 0:
        #     log.info('Test time ' + time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)) + ', ' + 'Accuracy: %d / %d\t%0.4f' % (correct_cnt, idx + 1, correct_cnt / (idx + 1)))

    log.info('Overall f1 score = %0.4f' % (f1_score(list(targets), list(predictions), average='weighted')))
    log.info('Overall accuracy = %0.2f%%' % (100 * correct_cnt / targets.shape[0]))
    return correct_cnt / targets.shape[0]
