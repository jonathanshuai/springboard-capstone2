import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch

class TransferModel(nn.Module):
    def __init__(self, base_model, n_classes):
        super(TransferModel, self).__init__()
        # Remove the fc layer
        self.base_layer = nn.Sequential(*list(base_model.children())[:-1])
    
        # Create our own fully connected layer
        self.fc = nn.Linear(base_model.fc.in_features, n_classes)
    
    def forward(self, inputs):
        # Connect the bottleneck layers with our fully connected layer 
        x = self.base_layer(inputs)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x