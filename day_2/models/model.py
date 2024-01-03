from torch import nn
from collections import OrderedDict


class MyAwesomeModel(nn.Module):
    """My awesome model."""    

    def __init__(self):
        super().__init__()

        # activation function
        self.relu = nn.ReLU()

        #layers
        self.fc1 = nn.Linear(784, 256)        
        self.fc2 = nn.Linear(256,128)        
        self.fc3 = nn.Linear(128, 64)        
        self.fc4 = nn.Linear(64, 10)
        self.logsoftmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        """Forward pass."""
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))        
        x = self.logsoftmax(self.fc4(x))
        return x
    
    




