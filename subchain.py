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
    def update(self, model):
        self.task.update(model)

class Owner:
    def __init__(self, name, dataloader):
        self.task = Task(dataloader)
        self.height = 0
        self.optimizer = optim.Adam(self.task.model.parameters(), lr=1e-2)
        self.name = name
        self.connected = []
    def connect(self, trainer):
        self.connected.append(trainer)
    def execute(self, predicted, labels):
        return self.task.backpropagation(self.optimizer, predicted, labels)
    def update(self, model):
        self.task.copyfrom(model)

class Subchain:
    def __init__(self, owner_size, trainer_size):
        dataloader = Dataloader('SST', 40, 50, 25000, (16,32,32))

        self.owners = []
        self.trainers = []
        for i in range(owner_size):
            self.owners.append(Owner(i, dataloader))
        for j in range(trainer_size):
            self.trainers.append(Trainer(j, dataloader))

    def start(self):
        # Global init
        global_model = self.owners[0].task.model
        global_height = 0
        
        for epoch in range(100):
            # Owner synchronize
            for owner in self.owners:
                owner.update(global_model)

            models = []
            aggr_weights = []
            for owner in self.owners:
                # Trainer init
                model = owner.task.model
                for trainer in self.trainers:
                    trainer.update(model)

                # Trainer predict
                predicts = []
                labels = []
                for trainer in self.trainers:
                    predicted, Y = trainer.execute()
                    predicts.append(predicted)
                    labels.append(Y)

                # Owner BP
                model, acc = owner.execute(torch.cat(predicts, 0), torch.cat(labels, 0))
                
                # Get weight
                weight = (global_height - owner.height + 1) ** -2
                models.append(model)
                aggr_weights.append(weight)

            # Models aggregate
            global_model.aggregate(models, aggr_weights)

            # Evaluate
            test_acc = self.owners[0].task.evaluate(global_model)
            print("epoch={},测试准确率={}".format(epoch, test_acc))