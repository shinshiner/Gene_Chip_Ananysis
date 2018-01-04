from __future__ import print_function, division
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data
from model import NET
from test import test
from torch.optim import Adam
import time
import random
from utils import setup_logger

parser = argparse.ArgumentParser(description='Gene-Chip-Classification')
parser.add_argument(
    '--train',
    default = True,
    metavar = 'T',
    help = 'train model (set False to evaluate)')
parser.add_argument(
    '--gpu',
    default=True,
    metavar='G',
    help='using GPU')
parser.add_argument(
    '--model-load',
    default=False,
    metavar='L',
    help='load trained model')
parser.add_argument(
    '--lr',
    type=float,
    default=0.00005,
    metavar='LR',
    help='learning rate')
parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S',
    help='random seed')
parser.add_argument(
    '--workers',
    type=int,
    default=1,
    metavar='W',
    help='how many training processes to use')
parser.add_argument(
    '--model-dir',
    type=str,
    default='trained_models/',
    metavar='MD',
    help='directory to store trained models')
parser.add_argument(
    '--log-dir',
    type=str,
    default='logs/',
    metavar='LD',
    help='directory to store logs')
parser.add_argument(
    '--epoch',
    type=int,
    default=0,
    metavar='EP',
    help='current epoch, used to pass parameters, do not change')
parser.add_argument(
    '--gamma',
    type=float,
    default=1,
    metavar='GM',
    help='to reduce learning rate gradually in simulated annealing')
parser.add_argument(
    '--batch_size',
    type=int,
    default=1,
    metavar='BS',
    help='the number of data to feed the nn in one time')
parser.add_argument(
    '--L2norm',
    default=True,
    metavar='L2',
    help='to use L2 norm or not')

dataset_path = "../output/data/dataset_test.npy"
target_path = "../output/data/target_test.npy"

if __name__ == '__main__':
    args = parser.parse_args()
    torch.set_default_tensor_type('torch.DoubleTensor')
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    
    if args.train:
        model = NET()
        if args.model_load:
            try:
                saved_state = torch.load(os.path.join(args.model_dir, 'best_model.dat'))
                model.load_state_dict(saved_state)
            except:
                print('Cannot load existing model from file!')
        if args.gpu:
            model = model.cuda()

        loss_func = nn.CrossEntropyLoss()
        dataset = torch.from_numpy(np.load("../output/data/dataset_train.npy"))
        targets = torch.from_numpy(np.int64(np.load("../output/data/target_train.npy")))
        dataset_test = np.load(dataset_path)
        targets_test = np.load(target_path)
        if args.L2norm:
            log_test = setup_logger(0, 'test_log_norm', os.path.join(args.log_dir, 'test_log_norm.txt'))
            log = setup_logger(0, 'train_log_norm', os.path.join(args.log_dir, 'train_log_norm.txt'))
            optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
        else:
            log_test = setup_logger(0, 'test_log', os.path.join(args.log_dir, 'test_log.txt'))
            log = setup_logger(0, 'train_log', os.path.join(args.log_dir, 'train_log.txt'))
            optimizer = Adam(model.parameters(), lr=args.lr)
        max_accuracy = 0.0

        # # code for batch training
        # torch_dataset = data.TensorDataset(data_tensor=dataset, target_tensor=targets)
        # data_loader = data.DataLoader(dataset=torch_dataset,
        #                               batch_size=args.batch_size,
        #                               shuffle=True,
        #                               num_workers=2,
        #                             )

        while True:
            args.epoch += 1
            print('=====> Train at epoch %d, Learning rate %0.6f <=====' % (args.epoch, args.lr))
            start_time = time.time()
            log.info('Train time ' + time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)) + ', ' + 'Training started.')

            # init
            order = list(range(targets.shape[0]))
            random.shuffle(order)
            losses = 0
            correct_cnt = 0

            # # code for batch training
            # for step, (batch_x, batch_y) in enumerate(data_loader):
            #     data_x = Variable(batch_x)
            #     data_y = Variable(batch_y)
            #     if args.gpu:
            #         data_x = data_x.cuda()
            #         data_y = data_y.cuda()
            #
            #     prediction = model(data_x)
            #     loss = loss_func(prediction, data_y)
            #     if args.gpu:
            #         loss = loss.cpu()
            #     optimizer.zero_grad()
            #     loss.backward()
            #     optimizer.step()

            for i in range(targets.shape[0]):
                idx = order[i]

                # get data
                data = Variable(dataset[idx])
                target = Variable(torch.LongTensor([targets[idx]]), requires_grad = False)

                if args.gpu:
                    data = data.cuda()
                    target = target.cuda()

                # get prediction and check it
                output = model(data)
                if args.gpu:
                    output = output.cpu()
                predict_class = output.max(0)[1].data.numpy()[0]
                if target.data[0] == predict_class:
                    correct_cnt += 1

                # update parameters
                optimizer.zero_grad()
                if args.gpu:
                    loss = loss_func(output.cuda().unsqueeze(0), target)
                else:
                    loss = loss_func(output.unsqueeze(0), target)
                loss.backward()
                if args.gpu:
                    loss = loss.cpu()
                optimizer.step()
                losses += loss

                # logging
                if (i + 1) % 500 == 0:
                    log.info('accuracy: %d%%' % (correct_cnt // 5))
                    correct_cnt = 0
                    log.info('Train time ' + \
                             time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)) + \
                             ', ' + 'Mean loss: %0.4f' % (losses.data.numpy()[0] / i))

            # save model
            if args.gpu:
                model = model.cpu()
            state_to_save = model.state_dict()
            if args.epoch % 10 == 0:
                torch.save(state_to_save, os.path.join(args.model_dir, 'epoch%d.dat' % args.epoch))
            accuracy = test(args, model, dataset_test, targets_test, log_test)
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                torch.save(state_to_save, os.path.join(args.model_dir, 'best_model.dat'))
            if args.gpu:
                model = model.cuda()

            # reduce learning rate
            args.lr *= args.gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
    else:
        #evaluate(args, os.path.join(Dataset_Dir, 'task2input.xml'), os.path.join(Dataset_Dir, 'task2output.xml'))
        pass
