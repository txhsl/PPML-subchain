import torch
import torch.nn as nn
import torch.optim as optim

from dataloader import Dataloader
from task import Task

class Trainer:
    def __init__(self, seq, dataloader):
        self.task = Task(dataloader, 0, 0)
        self.height = 0
        self.name = seq
        self.connected = []
    def connect(self, owner):
        self.connected.append(owner)
    def execute(self):
        return self.task.train()
    def update(self, model, height):
        self.task.update(model)
        self.height = height
    def run(self, batch_amount):
        for owner in self.connected:
            self.update(owner.task.model, owner.height)
            self.task.train_batches = owner.task.train_batches
            for batch in range(batch_amount):
                # Predict
                predicted, Y = self.execute()

                # Seedback
                owner.receive(predicted, Y, self.height)

class Owner:
    def __init__(self, seq, dataloader):
        self.task = Task(dataloader, seq*1000, (seq+1)*1000-1)
        self.height = 0
        self.optimizer = optim.Adam(self.task.model.parameters(), lr=1e-2)
        self.name = seq
        self.connected = []

        self.predicts = []
        self.labels = []
    def connect(self, trainer):
        self.connected.append(trainer)
    def execute(self, predicted, labels):
        return self.task.backpropagation(self.optimizer, predicted, labels)
    def update(self, model, height):
        self.task.copyfrom(model)
        self.height = height
    def receive(self, predicted, Y, height):
        if self.height == height:
            self.predicts.append(predicted)
            self.labels.append(Y)
    def start(self, chain, connections):
        # Build connection
        for connection in connections:
            self.connect(chain.trainers[connection])
            chain.trainers[connection].connect(self)

    def sync(self, chain):
        # Owner synchronize
        self.update(chain.model, chain.height)
        self.predicts.clear()
        self.labels.clear()

    def run(self, chain):
        # Collect from trainers

        # Owner BP
        model, acc = self.execute(torch.cat(self.predicts, 0), torch.cat(self.labels, 0))

        chain.block.append([model, self.height])

class Subchain:
    def __init__(self, owner_size, trainer_size):
        dataloader = Dataloader('SST', 40, 50, 25000, (16,32,32))
        self.height = 0
        self.block = []

        self.owners = []
        self.trainers = []
        for i in range(owner_size):
            self.owners.append(Owner(i, dataloader))
        for j in range(trainer_size):
            self.trainers.append(Trainer(j, dataloader))

        self.model = self.owners[0].task.model

    def run(self, connections):
        # FL settings
        alpha = 0.6
        model_amount = 4
        batch_amount = 32
        a = 3

        # Build network
        for idx in range(len(self.owners)):
            self.owners[idx].start(self, connections[idx])
        
        # Model training
        while self.height < 100:
            # Predict and BP
            for trainer in self.trainers:
                trainer.run(batch_amount)
            for owner in self.owners:
                owner.run(self)

            # Stage init
            models = []
            aggr_weights = []
            total_weight = 0

            # Aggregate model
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

            self.model.aggregate(models, aggr_weights)

            # Reset stage
            self.block.clear()
            self.height += 1
            models.clear()
            aggr_weights.clear()
            total_weight = 0

            # Broadcast result
            for owner in self.owners:
                owner.sync(self)

            # Evaluate
            test_acc = self.owners[0].task.evaluate(self.model)
            print("height={},测试准确率={}".format(self.height, test_acc))