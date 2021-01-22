import numpy as np
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from model import HETextCNN

def binary_acc(preds, y):
    correct = torch.eq(preds, y).float()
    acc = correct.sum() / len(correct)
    return acc

test_set_size = 16

class Task:

    def __init__(self, dataloader):
        # Text-CNN Parameter
        self.loader = dataloader

        sequence_length = 40
        vocab_size = self.loader.TEXT.vocab.vectors.shape[0]
        embedding_size = self.loader.TEXT.vocab.vectors.shape[1]
        num_classes = 2  # 0 or 1
        filter_sizes = [2, 3, 5] # n-gram window
        num_filters = 2

        self.train_batches = []
        for batch_idx , batch in enumerate(self.loader.train_iter):
            self.train_batches.append(batch)

        # pretrained embedding
        self.model = HETextCNN(num_filters, filter_sizes, vocab_size, embedding_size, sequence_length, num_classes)
        pretrained_embedding = self.loader.TEXT.vocab.vectors
        self.model.embedding.weight.data.copy_(pretrained_embedding)

    def update(self, model):
        self.model = model

    def train(self):
        self.model.train()

        batch = self.train_batches[random.randint(0, len(self.train_batches)-1)]
        text, labels = batch.text, batch.label - 1
        predicted = self.model(text)

        return predicted, labels

    def backpropagation(self, optimizer, predicted, labels):
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        
        acc = binary_acc(torch.max(predicted, dim=1)[1], labels)
        loss = criterion(predicted, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return self.model, acc

    def evaluate(self, model):
        avg_acc = []
        criterion = nn.CrossEntropyLoss()
        self.model.eval()

        for batch_idx , batch in enumerate(self.loader.dev_iter):
            if batch_idx >= test_set_size:
                continue
            text, labels = batch.text, batch.label - 1
            predicted = self.model(text)

            acc = binary_acc(torch.max(predicted, dim=1)[1], labels)
            avg_acc.append(acc)

        return np.array(avg_acc).mean()