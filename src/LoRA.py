import torch

class LoRA(torch.nn.Module):
    def __init__(self, linear_layer, rank=32, alpha=16):
        super(LoRA, self).__init__()
        ##### START CODE #####
        self.linear_layer = linear_layer
        in_dim = linear_layer.in_features
        std = 1 / torch.sqrt(torch.tensor(rank).float())
        self.adapter_downsample = torch.nn.Parameter(torch.randn(in_dim, rank) * std)
        self.adapter_upsample = torch.nn.Parameter(torch.zeros(rank, in_dim))
        self.adapter_alpha = alpha
        ##### END CODE #####

    def forward(self, x):
        ##### START CODE #####
        delta_x = self.adapter_alpha * (x @ self.adapter_downsample @ self.adapter_upsample)
        x = self.linear_layer(x) + delta_x
        return x
        ##### END CODE #####