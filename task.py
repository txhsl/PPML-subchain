import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from dataloader import Dataloader
from model import HETextCNN

def binary_acc(preds, y):
    correct = torch.eq(preds, y).float()
    acc = correct.sum() / len(correct)
    return acc

train_set_size = 128
test_set_size = 8

class Task:

    def __init__(self):
        # Text-CNN Parameter
        self.sequence_length = 40
        self.loader = Dataloader('SST', self.sequence_length, 50, 25000)

        train_batch_size = 8
        test_batch_size = 32

        self.vocab_size = self.loader.TEXT.vocab.vectors.shape[0]
        self.embedding_size = self.loader.TEXT.vocab.vectors.shape[1]
        self.num_classes = 2  # 0 or 1
        self.filter_sizes = [2, 3, 5] # n-gram window
        self.num_filters = 2
        self.lr = 1e-3

        self.train_iter, self.dev_iter, self.test_iter = self.loader.split(batch_sizes=(train_batch_size, test_batch_size, test_batch_size))

    def train(self, model, optimizer, criterion):
        avg_acc = []
        avg_loss = []
        model.train()
        for batch_idx , batch in enumerate(self.train_iter):
            if batch_idx >= train_set_size:
                continue
            text, labels = batch.text , batch.label - 1
            predicted = model(text)

            acc = binary_acc(torch.max(predicted, dim=1)[1], labels)
            avg_acc.append(acc)
            loss = criterion(predicted, labels)
            avg_loss.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return np.array(avg_acc).mean()

    def evaluate(self, model, criterion):
        avg_acc = []
        model.eval()
        for batch_idx , batch in enumerate(self.dev_iter):
            if batch_idx >= test_set_size:
                continue
            text, labels = batch.text , batch.label - 1
            predicted = model(text)

            acc = binary_acc(torch.max(predicted, dim=1)[1], labels)
            avg_acc.append(acc)

        return np.array(avg_acc).mean()

    def execute(self):
        model = HETextCNN(self.num_filters, self.filter_sizes, self.vocab_size, self.embedding_size, self.sequence_length, self.num_classes)
        print(model)

        pretrained_embedding = self.loader.TEXT.vocab.vectors
        model.embedding.weight.data.copy_(pretrained_embedding)

        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        train_accs, test_accs = [], []

        for epoch in range(50):

            train_acc = self.train(model, optimizer, criterion)
            print('epoch={},训练准确率={}'.format(epoch, train_acc))
            test_acc = self.evaluate(model, criterion)
            print("epoch={},测试准确率={}".format(epoch, test_acc))
            train_accs.append(train_acc)
            test_accs.append(test_acc)