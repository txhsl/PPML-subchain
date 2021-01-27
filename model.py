import numpy as np
import torch
import torch.nn as nn

class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
 
    def forward(self, x):
        x = 0.5 + 0.197*x - 0.004*torch.pow(x, 3)
        return x
    
class Softmax(nn.Module):
    def __init__(self):
        super(Softmax, self).__init__()
 
    def forward(self, x):
        x = 0.25 + 0.5*x - 0.125*torch.pow(x, 2)
        return x

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
 
    def forward(self, x):
        x = 0.1198 + 0.5*x + 0.1473*torch.pow(x, 2) - -0.002012*torch.pow(x, 4)
        return x
    
class HETextCNN(nn.Module):
    def __init__(self, num_filters, filter_sizes, vocab_size, embedding_size, sequence_length, num_classes):
        super(HETextCNN, self).__init__()

        self.sequence_length = sequence_length
        self.embedding_size = embedding_size
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.num_classes = num_classes

        self.num_filters_total = num_filters * len(filter_sizes)
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, num_filters, (kernel, embedding_size), bias=False),
                Swish(),
                nn.AvgPool2d((sequence_length - kernel + 1,1))
            ) for kernel in filter_sizes])
        
        self.fc = nn.Linear(self.num_filters_total, num_classes)
        self.sm = Softmax()
                           
    def forward(self, X):
        embedded_chars = self.embedding(X)# [batch_size, sequence_length, sequence_length]
        embedded_chars = embedded_chars.unsqueeze(1)

        out = [conv(embedded_chars) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.view(embedded_chars.size(0), -1)
        out = self.fc(out)
        logit = self.sm(out)
        return logit

    def aggregate(self, models, aggr_weights):
        conv_weights = []
        for size in self.filter_sizes:
            for channel in range(self.num_filters):
                conv_weights.append(np.zeros([size, self.embedding_size]))
        fc_weight = np.zeros(self.fc.weight.shape)
        fc_bias = np.zeros(self.fc.bias.shape)

        for i in range(len(models)):
            model, weight = models[i], aggr_weights[i]
            for j in range(len(model.convs)):
                conv_w = model.convs[j][0].weight.tolist()
                for channel in range(self.num_filters):
                    conv_weights[j * self.num_filters + channel] += np.array(conv_w[channel][0]) * weight

            fc_weight += np.array(model.fc.weight.tolist()) * weight
            fc_bias += np.array(model.fc.bias.tolist()) * weight

        for idx in range(len(self.convs)):
            self.convs[idx][0].weight.data.copy_(torch.from_numpy(np.array([[conv_weights[idx * self.num_filters]], [conv_weights[idx * self.num_filters + 1]]])))
        self.fc.weight.data.copy_(torch.from_numpy(fc_weight))
        self.fc.bias.data.copy_(torch.from_numpy(fc_bias))