# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
test_data = pd.read_csv("test.csv", delimiter=",", header=0)
train_data = pd.read_csv("train.csv", delimiter=",", header=0)
class DataSet:
    def __init__(self, data, flag):
        self.num_examples = len(data)
        self.pos = 0
        self.data = data
        if flag:
            self.images = np.array(data.iloc[:, 1:], dtype=np.float32)
        else:
            self.images = np.array(data.iloc[:, :], dtype=np.float32)
        self.images[self.images > 0] = 1
        if flag:
            self.labels = np.array(data.iloc[:, 0], dtype=np.int32)
            self.labels = self.one_hot(self.labels)
    def one_hot(self, labels):
        new_labels = []
        for x in labels:
            a = np.zeros(10, np.int32)
            a[x] = 1
            new_labels.append(a)
        return np.reshape(new_labels, newshape=(-1, 10))
    def next_batch(self, size):
        start = self.pos
        self.pos += size
        end = min(self.num_examples, self.pos)
        if end == self.num_examples:
            self.pos = 0
        return self.images[start:end, :], self.labels[start:end, :]

train = DataSet(train_data, True)
test =DataSet(test_data, False)