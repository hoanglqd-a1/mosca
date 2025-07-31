import torch
import torch.nn as nn
import torch.nn.functional as F

class ColorModule(nn.Module):
    def __init__(self, max_sh_degree):
        super().__init__()
        self.max_sh_degree = max_sh_degree
        self.input_dim = 3 * (max_sh_degree + 1) ** 2 + 6
        self.output_dim = 3 * (max_sh_degree + 1) ** 2
        self.linear1 = nn.Linear(self.input_dim, self.output_dim)
        self.reset_params()
    @classmethod
    def load_from_checkpoint(cls, ckpt):
        max_sh_degree = ckpt["max_sh_degree"]
        model = cls(max_sh_degree)
        model.load_state_dict(ckpt["state_dict"])
        return model
    def reset_params(self):
        torch.nn.init.normal_(self.linear1.weight, mean=0.0, std=2e-4)
        torch.nn.init.constant_(self.linear1.bias, 0.0)
        
    def forward(self, x):
        return x[:,:-6] + self.linear1(x)