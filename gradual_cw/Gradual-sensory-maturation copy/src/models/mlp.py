import torch.nn as nn

class Mlp(nn.Module):
    """Multi-layer perceptron model."""
    def __init__(self,
                 structure: list,
                 activation: str = 'relu',
                 initialization: str = "he",
                 bias: bool = True):
        """
        Initialize multi-layer perceptron model.

        Args:
            structure (list): List of integers representing the number of nodes in each layer.
                              e.g., [784, 128, 64, 10]
            activation (str): Activation function to use. Default is 'relu'.
                              e.g., 'relu', 'sigmoid', 'tanh'
            initialization (str): Initialization method to use. Default is 'he'.
                                  e.g., 'he', 'xavier'
            bias (bool): Whether to use bias. Default is True.
        """
        super(Mlp, self).__init__()
        self.structure = structure
        self.linear = []
        self.act = []

        for i in range(len(structure) - 1):
            self.linear.append(nn.Linear(self.structure[i], structure[i + 1], bias=bias))
            match activation:
                case 'relu':
                    self.act.append(nn.ReLU())
                case 'sigmoid':
                    self.act.append(nn.Sigmoid())
                case 'tanh':
                    self.act.append(nn.Tanh())

        self.linear = nn.ModuleList(self.linear)
        self.act = nn.ModuleList(self.act)
        self.initialize_weights(initialization, activation)

    def initialize_weights(self, initialization, activation):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if initialization == 'he':
                    nn.init.kaiming_normal_(m.weight, nonlinearity=activation)
                elif initialization == 'xavier':
                    nn.init.xavier_normal_(m.weight)

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for idx in range(len(self.linear) - 1):
            x = self.linear[idx](x)
            x = self.act[idx](x)
        x = self.linear[-1](x)
        return x
