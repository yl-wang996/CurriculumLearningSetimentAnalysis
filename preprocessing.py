# -*- coding: utf-8 -*-
# @Time    : 2021/11/23
# @Author  : Yuqian Lei & Yunlong Wang
import os
import tqdm
import numpy as np

import random

data_root = '/data/1wang/data/CLinSematicClassification/dataset'
org_dt = os.path.join(data_root,'aclImdb')
from pathlib import Path

def dataset2npz():
    def read_imdb_split(split_dir):
        split_dir = Path(split_dir)
        texts = []
        labels = []
        for label_dir in ["pos", "neg"]:
            tq_iter = tqdm.tqdm((split_dir / label_dir).iterdir())
            for text_file in tq_iter:
                texts.append(text_file.read_text())
                labels.append(0 if label_dir == "neg" else 1)
        return texts, labels

    train_texts, train_labels = read_imdb_split(os.path.join(org_dt, 'train'))
    test_texts, test_labels = read_imdb_split(os.path.join(org_dt, 'test'))
    np.savez(os.path.join(data_root,'orign_dataset.npz'),
             train_texts=train_texts,
             train_labels=train_labels,
             test_texts=test_texts,
             test_labels=test_labels)

def load_data(path):
    with open(path,'rb') as f:
        data = np.load(f)
        train_texts = data['train_texts']
        train_labels = data['train_labels']
        test_texts = data['test_texts']
        test_labels = data['test_labels']
    return train_texts,train_labels,test_texts,test_labels


def shuffle(data,label):
    random.seed(2021)
    ids = [n for n in range(len(data))]
    random.shuffle(ids)
    data = data[ids]
    label = label[ids]
    return data,label



def generate_subset(train_num,test_num):
    # totally, trainset has 25000 examples as same as testset
    # we pick 2500 for training and 500 for testing to accelerate the calculation
    train_texts,train_labels,test_texts,test_labels = load_data(os.path.join(data_root,'orign_dataset.npz'))
    train_texts, train_labels = shuffle(train_texts,train_labels)
    test_texts, test_labels = shuffle(test_texts,test_labels)
    small_train_texts = train_texts[:train_num]
    small_train_labels = train_labels[:train_num]
    small_test_texts = test_texts[:test_num]
    small_test_labels = test_labels[:test_num]
    np.savez(os.path.join(data_root,f'Subset_tr{train_num}_test{test_num}_shuffled.npz'),
             train_texts=small_train_texts,
             train_labels=small_train_labels,
             test_texts=small_test_texts,
             test_labels=small_test_labels)

if __name__ == '__main__':
    # dataset2npz()
    generate_subset(train_num=2500,test_num=500)