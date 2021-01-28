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
    def start(self):
        while True:
            for owner in self.connected:
                self.update(owner.task.model, owner.height)

                # Predict
                owner.model_lock.acquire()
                predicted, Y = self.execute()
                owner.model_lock.release()

                # Seedback
                owner.message_lock.acquire()
                owner.receive(predicted, Y, self.height)
                owner.message_lock.release()

class Owner:
    def __init__(self, name, dataloader):
        self.task = Task(dataloader)
        self.height = 0
        self.optimizer = optim.Adam(self.task.model.parameters(), lr=1e-2)
        self.name = name
        self.connected = []

        self.predicts = []
        self.labels = []
        self.message_lock = _thread.allocate_lock()
        self.model_lock = _thread.allocate_lock()
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
    def start(self, chain, connections, batch_amount):
        for connection in connections:
            self.connect(chain.trainers[connection])
            chain.trainers[connection].connect(self)
        
        for epoch in range(100):
            # Owner synchronize
            self.update(chain.model, chain.height)

            # Collect from trainers
            while len(self.predicts) < batch_amount:
                time.sleep(1)

            # Owner BP
            self.model_lock.acquire()
            self.message_lock.acquire()

            model, acc = self.execute(torch.cat(self.predicts, 0), torch.cat(self.labels, 0))
            self.predicts.clear()
            self.labels.clear()

            self.message_lock.release()
            self.model_lock.release()

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
        model_amount = 8
        batch_amount = 32
        a = 10

        for idx in range(len(self.owners)):
            _thread.start_new_thread(self.owners[idx].start, (self, connections[idx], batch_amount))
        
        for trainer in self.trainers:
            _thread.start_new_thread(trainer.start, ())

        time.sleep(0.5)
        # Models aggregate
        while True:
            while len(self.block) < model_amount:
                time.sleep(1)

            # Stage init
            models = []
            aggr_weights = []
            total_weight = 0

            # Build block
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

            # Reset stage
            self.block.clear()
            self.height += 1
            models.clear()
            aggr_weights.clear()
            total_weight = 0

            # Evaluate
            test_acc = self.owners[0].task.evaluate(self.model)
            print("epoch={},测试准确率={}".format(self.height, test_acc))