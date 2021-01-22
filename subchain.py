from task import Task
import torch
import torch.nn as nn
import torch.optim as optim

from dataloader import Dataloader

class Trainer:
    def __init__(self, name, dataloader):
        self.task = Task(dataloader)
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
        self.name = name
        self.connected = []
    def connect(self, trainer):
        self.connected.append(trainer)
    def execute(self, optimizer, predicted, labels):
        return self.task.backpropagation(optimizer, predicted, labels)
    def update(self, model):
        self.task.update(model)

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
        # Temp global init
        model = self.owners[0].task.model
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        
        for epoch in range(100):
            models = []
            
            for owner in self.owners:
                # Trainer init
                model = owner.task.model
                for trainer in self.trainers:
                    trainer.update(model)

                predicted = []
                labels = []
                for trainer in self.trainers:
                    predicts, Y = trainer.execute()
                    predicted.append(predicts)
                    labels.append(Y)

                model, acc = owner.execute(optimizer, torch.cat(predicted, 0), torch.cat(labels, 0))
                models.append(model)

            model.aggregate(models)

            # Owner init
            for owner in self.owners:
                owner.update(model)

            test_acc = self.owners[0].task.evaluate(model)
            print("epoch={},测试准确率={}".format(epoch, test_acc))