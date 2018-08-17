import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Combined(nn.Module):
    def __init__(self, base_model, n_classes):
        super(Combined, self).__init__()
        self.base_layer = nn.Sequential(*list(base_model.children())[:-1])

        # Remove the fc layer
        # self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(base_model.fc.in_features, n_classes)

    def forward(self, inputs):
        x = self.base_layer(inputs)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x

