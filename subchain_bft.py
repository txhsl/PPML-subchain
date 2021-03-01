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
    def start(self, chain, node, connections):
        # Build connection
        for connection in connections:
            self.connect(chain.trainers[connection])
            chain.trainers[connection].connect(self)
        
        # Owner synchronize
        self.update(chain.model, chain.height)
        node.model = self.task.model

    def run(self, chain, node, a, b):
        # Collect from trainers

        # Owner BP
        model, acc = self.execute(torch.cat(self.predicts, 0), torch.cat(self.labels, 0))

        # Models aggregate
        models = []
        aggr_weights = []
        total_weight = 0
        for peer in chain.bft_nodes:
            models.append(peer.model)
            weight = (self.height - peer.model_seq + 1) ** -a
            aggr_weights.append(weight)
            total_weight += weight
        for idx in range(len(aggr_weights)):
            aggr_weights[idx] /= total_weight
            aggr_weights[idx] *= 1 - b

        models.append(model)
        aggr_weights.append(b)
        print(aggr_weights)
        model.aggregate(models, aggr_weights)

        # Node update
        #node.model = self.task.model
        node.model_seq = self.height

        self.predicts.clear()
        self.labels.clear()

    def evaluate(self):
        return self.task.evaluate(self.task.model)

class Subchain:
    def __init__(self, owner_size, trainer_size):
        dataloader = Dataloader('SST', 40, 50, 25000, (16,32,32))
        self.height = 0
        self.block = []
        self.ledger = []
        self.bft_nodes = []

        self.owners = []
        self.trainers = []
        for i in range(owner_size):
            self.owners.append(Owner(i, dataloader))
            self.bft_nodes.append(Node(i, owner_size))
        for j in range(trainer_size):
            self.trainers.append(Trainer(j, dataloader))

        self.model = self.owners[0].task.model

    def run(self, connections):
        # FL settings
        batch_amount = 32
        a = 3
        b = 0.6

        # Build network
        for node in self.bft_nodes:
            node.peers = self.bft_nodes

        for idx in range(len(self.owners)):
            owner = self.owners[idx]
            node = self.bft_nodes[idx]
            owner.start(self, node, connections[idx])

        # BFT-based Training
        while self.height < 200:
            heights = []
            # Foreach
            for idx in range(len(self.bft_nodes)):
                node = self.bft_nodes[idx]
                owner = self.owners[idx]

                # Train
                if node.is_primary():
                    if node.lock and node.model_seq != node.height:
                        node.lock = False
                    if not node.lock:
                        # Predict
                        for trainer in self.trainers:
                            trainer.run(batch_amount)
                        # BP and aggregate
                        owner.run(self, node, a, b)
                        # Evaluate
                        test_acc = owner.evaluate()
                        print("height={},node={},测试准确率={}".format(owner.height, idx, test_acc))
                        # Mark as trained
                        node.lock = True

                # Push bft
                node.run()
                
                owner.height = node.height
                heights.append(node.height)

            self.height = max(heights)