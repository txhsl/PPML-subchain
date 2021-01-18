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
