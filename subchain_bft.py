import torch
import torch.nn as nn
import torch.optim as optim

from dataloader import Dataloader
from task import Task
from bft import State, Node

class Trainer:
    def __init__(self, seq, dataloader):
        self.task = Task(dataloader, 0, 0)
        self.height = 0
        self.seq = seq
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
    def __init__(self, seq, net_size, dataloader):
        self.task = Task(dataloader, seq*1000, (seq+1)*1000-1)
        self.height = 0
        self.optimizer = optim.Adam(self.task.model.parameters(), lr=1e-2)
        self.seq = seq
        self.connected = []
        self.node = Node(seq, net_size)

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
        
        # Owner synchronize
        self.update(chain.model, chain.height)
        self.node.model = self.task.model

    def run(self, network, a, b, gap):
        # Collect from trainers

        # Owner BP
        model, acc = self.execute(torch.cat(self.predicts, 0), torch.cat(self.labels, 0))

        # Models aggregate
        models = []
        aggr_weights = []
        total_weight = 0
        for peer in network:
            models.append(peer.model)
            weight = 1
            aggr_weights.append(weight)
            total_weight += weight
        for idx in range(len(aggr_weights)):
            if total_weight == 0:
                aggr_weights[idx] = 0
            else:
                aggr_weights[idx] /= total_weight
                aggr_weights[idx] *= 1 - b * gap ** -a

        models.append(model)
        aggr_weights.append(b * gap ** -a)
        print(aggr_weights)
        self.aggregate(models, aggr_weights)

        # Node update
        self.node.model = self.task.model
        self.node.model_seq = self.height

        self.predicts.clear()
        self.labels.clear()
    def aggregate(self, models, weights):
        self.task.model.aggregate(models, weights)
    def evaluate(self):
        return self.task.evaluate(self.task.model)

class Subchain:
    def __init__(self, owner_size, trainer_size):
        dataloader = Dataloader('SST', 40, 50, 25000, (16,32,32))
        self.height = 0
        self.block = []
        self.ledger = []

        self.owners = []
        self.trainers = []
        for i in range(owner_size):
            self.owners.append(Owner(i, owner_size, dataloader))
        for j in range(trainer_size):
            self.trainers.append(Trainer(j, dataloader))

        self.model = self.owners[0].task.model

    def run(self, connections):
        # FL settings
        batch_amount = 32
        a = 0.5
        b = 0.8

        # Build network
        nodes = []
        for owner in self.owners:
            nodes.append(owner.node)
        for owner in self.owners:
            owner.node.peers = nodes

        for idx in range(len(self.owners)):
            owner = self.owners[idx]
            owner.start(self, connections[idx])

        # BFT-based Training
        while self.height < 200:
            heights = []
            # Foreach
            for idx in range(len(self.owners)):
                owner = self.owners[idx]
                node = owner.node

                # Train
                if owner.node.is_primary():
                    if owner.node.is_locked() and owner.node.model_seq != owner.node.height:
                        owner.node.release()
                    if not owner.node.is_locked():
                        gap = 1 + 1
                        base_model_seq = 0
                        if owner.seq < gap:
                            base_model_seq = owner.seq - gap + len(self.owners)
                        else:
                            base_model_seq = owner.seq - gap
                        owner.update(self.owners[base_model_seq].task.model, self.height)
                        # Predict
                        for trainer in self.trainers:
                            trainer.run(batch_amount)
                        # BP and aggregate
                        owner.run(nodes, a, b, gap)
                        # Evaluate
                        test_acc = owner.evaluate()
                        print("height={},node={},测试准确率={}".format(owner.height, idx, test_acc))
                        # Mark as trained
                        owner.node.lock()

                # Push bft
                owner.node.run()
                
                owner.height = owner.node.height
                heights.append(owner.height)

            self.height = max(heights)