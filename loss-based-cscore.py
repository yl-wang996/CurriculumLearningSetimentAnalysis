# -*- coding: utf-8 -*-
# @Time    : 2021/11/23
# @Author  : Yuqian Lei & Yunlong Wang
from utils.data import MyDataset
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification
import torchmetrics
from transformers import DistilBertConfig
import random
from utils.utils import  get_optimizer, set_seed
from utils.dataset import Subset
from torch import nn
import os
import tqdm
import argparse
import collections
from utils.earlystop import EarlyStopping
import time
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

def set_args():
    parser = argparse.ArgumentParser(description='PyTorch Training')

    parser.add_argument('--dataset', default='dataset',type=str,
                        help='path to dataset (default: dataset)')
    # parser.add_argument('--arch', metavar='ARCH', default='resnet18',
    #                     help='model architecture: (default: resnet18)')
    # parser.add_argument('--dataset', default='cifar10', type=str,
    #                     help='dataset')
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=1000, type=int,
                        help='number of total epochs to run')
    # parser.add_argument('--epochs', default=30, type=int,
    #                     help='number of total epochs to run')
    parser.add_argument('--batch_size', default=10, type=int,
                        help='mini-batch size (default: 512), this is the total')
    parser.add_argument('--optimizer', default="sgd", type=str,
                        help='optimizer')
    parser.add_argument('--num_runs', default=4, type=int,
                        help='checkpoint model to resume')
    parser.add_argument('--scheduler', default="cosine", type=str,
                        help='lr scheduler')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--printfreq', default=20, type=int,
                        help='print frequency (default: 10)')
    # parser.add_argument('--evaluate', action='store_true',
    #                     help='evaluate model on validation set')
    parser.add_argument('--seed', default=2021, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--logdir', default='logdir', type=str,
                        help='prefix to use when saving files')
    parser.add_argument('--rand_fraction', default=0., type=float,
                        help='label curruption (default:0)')
    # parser.add_argument('--send_email',  action='store_true',
    #                     help='send the message after training')
    parser.add_argument('--half',  action='store_true',
                        help='acclerate the computing')
    # parser.add_argument('--gpu_id', default='0', type=str,
    #                     help='setup the id of used GPU')

    args = parser.parse_args()
    return args

class MyTrainer():
    def __init__(self, args, train_loader, val_loader):
        self.args = args
        self.train_loader=train_loader
        self.val_loader=val_loader
        self.model = self.build_model()
        self.optimizer = get_optimizer(
            optimizer_name=args.optimizer,
            parameters=self.model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.wd)
        # self.scheduler = get_scheduler(
        #     scheduler_name=config.scheduler_name,
        #     optimizer=self.optimizer,
        #     num_epochs=config.epochs)

        self.loss_fn = nn.CrossEntropyLoss(reduction="none").cuda()
        self.acc_metric = torchmetrics.Accuracy()
        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "iter": [0, ]}


    def build_model(self,pretraining=True):
        # init model
        if pretraining:
            # load the model with pretaining.
            model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
        else:
            # load the model without pretaining.
            distilbert_config = DistilBertConfig()
            model = DistilBertForSequenceClassification(distilbert_config)
        if self.args.half == True:
            print('Using half precision except in Batch Normalization!')
            model = model.half()
            for module in model.modules():
                if (isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d)):
                    module.float()
        model.to(self.args.device)
        return model

    def subset_train(self,seed,val_order,ind_loss):
        self.model.train()
        set_seed(seed)
        start_epoch = 0
        es = EarlyStopping(patience=5, mode='max')
        for epoch in range(start_epoch, start_epoch + self.args.epochs):
            loss_acc = 0
            tr_acc = 0
            iter = 0
            tq_loader = tqdm.tqdm(self.train_loader,f'subset training at epoch {epoch}/{self.args.epochs}')
            for i, batch in enumerate(tq_loader):
                sentences = batch['input_ids'].to(self.args.device)
                attention_mask = batch['attention_mask'].to(self.args.device)
                labels = batch['labels'].to(self.args.device)
                outputs = self.model(sentences, attention_mask=attention_mask, labels=labels)
                loss = outputs['loss']
                logits = outputs['logits']
                labels = labels.to('cpu')
                confidence = logits.softmax(dim=-1).to('cpu')
                tr_acc +=  self.acc_metric(confidence, labels)
                iter+=1

                loss_acc += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print(f'train at epoch {epoch}/{self.args.epochs} with loss:{loss_acc/iter:.4f}, acc:{tr_acc/iter:.4f}')
            ind_loss,val_loss,val_acc = self.validate(val_order, ind_loss)
            # print(f'!!!!!!!!ACC:{val_acc}')
            if es.step(val_acc):
                print(f'earlt stop at epoch:{epoch}, val_loss:{val_loss} val_acc:{val_acc:.4f}')
                break
        return ind_loss

    def validate(self,val_order, ind_loss):
        # switch to evaluate mode
        self.model.eval()
        start = 0
        loss_acc = 0
        acc = 0
        iter = 0
        with torch.no_grad():

            for i, batch in enumerate(val_loader):
                sentences = batch['input_ids'].to(self.args.device)
                attention_mask = batch['attention_mask'].to(self.args.device)
                labels = batch['labels'].to(self.args.device)
                outputs = self.model(sentences, attention_mask=attention_mask, labels=labels)
                loss = outputs['loss'].item()
                logits = outputs['logits'].to(self.args.device)
                indloss = self.loss_fn(logits, labels)

                # compute metrcs
                labels = labels.to('cpu')
                confidence = logits.softmax(dim=-1).to('cpu')
                acc += self.acc_metric(confidence, labels)
                iter += 1


                list(map(lambda a, b: ind_loss[a].append(b), val_order[start:start + len(labels)], indloss))
                start += len(labels)
                loss_acc += torch.sum(indloss).item()
            print(f'validation with loss:{loss_acc/iter:.4f} acc:{acc/iter:.4f}')
        return ind_loss,loss,acc/iter

if __name__ == '__main__':
    args = set_args()
    args.dataset = '/data/1wang/data/CLinSematicClassification/dataset/Subset_tr2500_test500_shuffled.npz'
    args.rand_fraction = 0.4
    args.logdir = f'/data/1wang/data/CLinSematicClassification/Logdir4LossScore'
    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)
    assert os.path.exists(args.dataset)
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')


    dataset = MyDataset(data_path=args.dataset)
    # dataset = MyDataset(data_path='origin_100_sub_dataset_shuffled.npz')
    # if rand_fraction equal 0, the train_dataset_N as same as train_dataset.
    train_dataset_clean,train_dataset,test_dataset = dataset.get_corrupt_dataset(args.rand_fraction)

    if args.rand_fraction == 0.0:
        name = 'IMDB'
    else:
        name = f'IMDB_N{args.rand_fraction}'

    order = [i for i in range(len(train_dataset))]
    ind_loss = collections.defaultdict(list)
    for i_run in range(args.num_runs):
        random.shuffle(order)
        startIter = 0
        # split data into 4 pieces, one for val, three for train
        for i in range(4):
            print(f'-----------i_run:{i_run}----------piece:{i}------------------')
            if i==3:
                startIter_next = len(train_dataset)
            else:
                startIter_next = int(startIter+1/4*len(train_dataset))
            print(f'i_run {i_run} and order  =============> from {startIter} to {startIter_next}')
            val_subsets = Subset(train_dataset_clean, list(order[startIter:startIter_next]))
            train_subsets = Subset(train_dataset, list(order[0:startIter]) + list(order[startIter_next:]))

            train_loader = torch.utils.data.DataLoader(train_subsets, batch_size=args.batch_size,
                                                       shuffle=True, num_workers=args.workers, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(val_subsets, batch_size=args.batch_size,
                                                     shuffle=False, num_workers=args.workers, pin_memory=True)
            trainer = MyTrainer(args=args,
                                train_loader=train_loader,
                                val_loader=val_loader)
            val_order = list(order[startIter:startIter_next])
            ind_loss = trainer.subset_train(seed=i,
                                 val_order=val_order,
                                 ind_loss=ind_loss)

            startIter += int(1 / 4 * len(train_dataset))
            del train_loader
            del val_loader
            del trainer

        stat = {k: [torch.mean(torch.tensor(v)), torch.std(torch.tensor(v))] for k, v in
                sorted(ind_loss.items(), key=lambda item: sum(item[1]))}
        if i_run == args.num_runs - 1:
            torch.save(stat, os.path.join(args.logdir, f'{name}.order.pth'))
        else:
            torch.save(stat, os.path.join(args.logdir, f'{name}.order_{i_run}.pth'))
