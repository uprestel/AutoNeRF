import torch
import torch.nn as nn
import functools


class BasicFullyConnectedNet(nn.Module):
    """
        This class implements the architectures used in s and t.
    """
    def __init__(self, dim, depth, hidden_dim=256, use_tanh=False, use_bn=False, out_dim=None):
        """
            dim: The input dimension of our network
            depth: The depth of our network
            hidden_dim: The hidden dimension of our network
            use_tanh: boolean to specify if we want a boolean at the end.
            use_bn: boolean to specify if we want batchnorm layers in between.
            out_dim: the output dimension of the network
        """
        super(BasicFullyConnectedNet, self).__init__()

        # we define our layers
        layers = []
        layers.append(nn.Linear(dim, hidden_dim))

        # Add batch norm layer?
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.LeakyReLU())

        # Add layers
        for d in range(depth):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(hidden_dim, dim if out_dim is None else out_dim))

        # Add optional tanh activation at the end
        if use_tanh:
            layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)


    def forward(self, x):
        print("in BFCN", self.main)
        z = x
        for layer in self.main:
            print("at", layer, "with", x.shape)
            z = layer(z)

        return z
        #return self.main(x)


#--------------------------------------------------------------------------------------------------------------------
class ConditionalDoubleVectorCouplingBlock(nn.Module):
    """
        This implements s_theta and t_theta as defined in the paper.
    """
    def __init__(self, in_channels, cond_channels, hidden_dim, depth=2):
        """
            in_channels: the size of the input. This should be divisible by 2.
            cond_channels: the size of the conditional H(y).
        """
        
        super(ConditionalDoubleVectorCouplingBlock, self).__init__()
        
        
        # since we split the input into two halves, we only feed s and t respectively 
        # the input size in_channels // 2 + cond_channels.
        
        # now we define our two pairs of coupling networks.
        self.s = nn.ModuleList([
            BasicFullyConnectedNet(dim=in_channels // 2 + cond_channels, depth=depth,
                                   hidden_dim=hidden_dim, use_tanh=True,
                                   out_dim=in_channels // 2) for _ in range(2)])
        self.t = nn.ModuleList([
            BasicFullyConnectedNet(dim=in_channels // 2 + cond_channels, depth=depth,
                                   hidden_dim=hidden_dim, use_tanh=False,
                                   out_dim=in_channels // 2) for _ in range(2)])


    def forward(self, x, x_cond, reverse=False):
