import _thread
import time
import torch
import torch.nn as nn
import torch.optim as optim

from dataloader import Dataloader
from task import Task

class Trainer:
    def __init__(self, name, dataloader):
        self.task = Task(dataloader)
        self.height = 0
        self.name = name
        self.connected = []
    def connect(self, owner):
        self.connected.append(owner)
    def execute(self):
        return self.task.train()
    def update(self, model, height):
        self.task.update(model)
        self.height = height

class Owner:
    def __init__(self, name, dataloader):
        self.task = Task(dataloader)
        self.height = 0
        self.optimizer = optim.Adam(self.task.model.parameters(), lr=1e-2)
        self.name = name
        self.duration = 3
        self.connected = []
    def connect(self, trainer):
        self.connected.append(trainer)
    def execute(self, predicted, labels):
        return self.task.backpropagation(self.optimizer, predicted, labels)
    def update(self, model, height):
        self.task.copyfrom(model)
        self.height = height

class Subchain:
    def __init__(self, owner_size, trainer_size):
        dataloader = Dataloader('SST', 40, 50, 25000, (16,32,32))
        self.height = 0
        self.block = []
        self.blocktime = 1
        self.owners = []
        self.trainers = []
        for i in range(owner_size):
            self.owners.append(Owner(i, dataloader))
        for j in range(trainer_size):
            self.trainers.append(Trainer(j, dataloader))

        self.model = self.owners[0].task.model

    def groupstart(self, owner, connections):
        trainers = []
        for connection in connections:
            trainers.append(self.trainers[connection])
        
        for epoch in range(100):
            # Owner synchronize
            owner.update(self.model, self.height)

            # Trainer init
            model = owner.task.model
            for trainer in trainers:
                trainer.update(model, owner.height)

            # Trainer predict
            predicts = []
            labels = []
            for trainer in trainers:
                predicted, Y = trainer.execute()
                predicts.append(predicted)
                labels.append(Y)

            # Time cost
            time.sleep(self.blocktime)

            # Owner BP
            model, acc = owner.execute(torch.cat(predicts, 0), torch.cat(labels, 0))
                
            self.block.append([model, owner.height])

    def run(self, connections):
        # FL settings
        alpha = 0.6
        a = 10

        for idx in range(len(self.owners)):
            _thread.start_new_thread(self.groupstart, (self.owners[idx], connections[idx]))

        time.sleep(0.5)
        # Models aggregate
        while True:
            time.sleep(self.blocktime)

            models = []
            aggr_weights = []
            total_weight = 0
            if len(self.block) == 0:
                continue
            print("Height", self.height + 1, "Blocksize", len(self.block))
            for transaction in self.block:
                models.append(transaction[0])
                weight = (self.height - transaction[1] + 1) ** -a
                aggr_weights.append(weight)
                total_weight += weight
            for idx in range(len(aggr_weights)):
                aggr_weights[idx] /= total_weight
                aggr_weights[idx] *= 1 - alpha
            models.append(self.model)
            aggr_weights.append(alpha)
            print(aggr_weights)
            self.model.aggregate(models, aggr_weights)

            self.block.clear()
            self.height += 1
            models.clear()
            aggr_weights.clear()
            total_weight = 0

            # Evaluate
            test_acc = self.owners[0].task.evaluate(self.model)
            print("epoch={},测试准确率={}".format(self.height, test_acc))