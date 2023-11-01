# -*- coding: utf-8 -*-
# @Time    : 2021/11/28
# @Author  : Yuqian Lei & Yunlong Wang
import numpy as np
import torch
from transformers import DistilBertTokenizerFast
import bisect

# for pytorch dataloader
class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['idx'] = idx
        return item

    def __len__(self):
        return len(self.labels)

# prepare data
class MyDataset():
    def __init__(self,data_path):
        self.data_path = data_path
        with open(data_path,'rb') as f:
            data = np.load(f)
            self.train_texts = data['train_texts']
            self.train_labels = data['train_labels']
            self.test_texts = data['test_texts']
            self.test_labels = data['test_labels']
            self.tokenizer_name = 'distilbert-base-uncased'
            self.train_len = len(data['train_labels'])
            self.test_len = len(data['test_labels'])

    def get_dataset(self):
        # tokenization
        tokenizer = DistilBertTokenizerFast.from_pretrained(self.tokenizer_name)
        train_encodings = tokenizer(list(self.train_texts), truncation=True, padding=True)
        test_encodings = tokenizer(list(self.test_texts), truncation=True, padding=True)

        # init Dataset class for pytorch
        train_dataset = IMDbDataset(train_encodings, self.train_labels)
        test_dataset = IMDbDataset(test_encodings, self.test_labels)

        return train_dataset,test_dataset

    def get_corrupt_dataset(self, rand_fraction=0.0,save = False):
        # tokenization
        tokenizer = DistilBertTokenizerFast.from_pretrained(self.tokenizer_name)
        train_encodings = tokenizer(list(self.train_texts), truncation=True, padding=True)
        test_encodings = tokenizer(list(self.test_texts), truncation=True, padding=True)

        # init Dataset class for pytorch
        train_dataset_clean = IMDbDataset(train_encodings, self.train_labels)
        test_dataset = IMDbDataset(test_encodings, self.test_labels)
        assert rand_fraction >= 0.0 and rand_fraction <= 1.0
        if rand_fraction > 0.0:
            train_labels_N = self.corrupt_fraction_of_data(self.train_labels,rand_fraction)
            train_dataset = IMDbDataset(train_encodings, train_labels_N)
            if save:
                np.savez(f'{self.data_path[:-4]}_N{rand_fraction}.npz',
                         train_texts=self.train_texts,
                         train_labels=train_labels_N,
                         test_texts=self.test_texts,
                         test_labels=self.test_labels)
        else:
            train_dataset = train_dataset_clean
        return train_dataset_clean,train_dataset,test_dataset


    def corrupt_fraction_of_data(self,labels,rand_fraction):
        """Corrupts fraction of train data by permuting image-label pairs."""
        length = len(labels) // 4
        start_next = 0
        new_labels = []
        for _ in range(4):
            nr_corrupt_instances = start_next + int(np.floor(length * rand_fraction))
            print('Randomizing {} fraction of data == {} / {}'.format(rand_fraction,
                                                                      nr_corrupt_instances - start_next,
                                                                      length))
            # We will corrupt the top fraction data points
            corrupt_label = labels[start_next:nr_corrupt_instances]
            clean_label = labels[nr_corrupt_instances:start_next + length]

            # Corrupting data
            np.random.seed(2021)
            rand_idx = np.random.permutation(np.arange(start_next, nr_corrupt_instances))
            corrupt_label = np.array(corrupt_label)[rand_idx - start_next]
            # Adding corrupt and clean data back together
            new_labels.extend(corrupt_label)
            new_labels.extend(clean_label)
            start_next += length

        return np.array(new_labels)