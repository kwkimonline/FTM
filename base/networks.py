import torch.nn as nn


class MLP(nn.Module):
    
    def __init__(self, input_dim, hidden_dims=[200, 500], output_act=None, output_dim=None, act='ReLU', bn=False, dropout=False):
        super(MLP, self).__init__()
        output_dim = input_dim if output_dim is None else output_dim
        
        if len(hidden_dims) > 0:
            network_modules = [nn.Linear(input_dim, hidden_dims[0])]
            if act != 'LeakyReLU':
                network_modules.append(getattr(nn, act)())
            else:
                network_modules.append(getattr(nn, act)(negative_slope=0.2, inplace=True))
            for i in range(len(hidden_dims) - 1):
                network_modules.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
                if bn:
                    network_modules.append(nn.BatchNorm1d(hidden_dims[i + 1], affine=False))
                if dropout:
                    network_modules.append(nn.Dropout())
                if act != 'LeakyReLU':
                    network_modules.append(getattr(nn, act)())
                else:
                    network_modules.append(getattr(nn, act)(negative_slope=0.2, inplace=True))
            network_modules.append(nn.Linear(hidden_dims[-1], hidden_dims[-1]))
            self.extractor = nn.Sequential(*network_modules)
            self.final_layer = nn.Linear(hidden_dims[-1], output_dim)
            if output_act != None:
                self.final_layer = nn.Sequential(nn.Linear(hidden_dims[-1], output_dim),
                                                 getattr(nn, output_act)())
            # network_modules.append(self.final_layer)
            self.network = nn.Sequential(self.extractor, self.final_layer)
        else:
            network_modules = [nn.Linear(input_dim, output_dim)]
            self.network = nn.Sequential(*network_modules)

    def forward(self, x, extract=False):
        return self.network(x) if not extract else self.extractor(x)

    def freeze(self):
        for param in self.network.parameters():
            param.requires_grad = False
            
    def melt(self):
        for param in self.network.parameters():
            param.requires_grad = True