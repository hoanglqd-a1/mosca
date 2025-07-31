import torch
import torch.nn as nn
import torch.nn.functional as F

class ColorModule(nn.Module):
    def __init__(self, max_sh_degree):
        super().__init__()
        self.max_sh_degree = max_sh_degree
        self.input_dim = 3 * (max_sh_degree + 1) ** 2 + 6
        self.output_dim = 3 * (max_sh_degree + 1) ** 2
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_dim)
        )
        self.reset_params()
    @classmethod
    def load_from_checkpoint(cls, ckpt):
        max_sh_degree = ckpt["max_sh_degree"]
        model = cls(max_sh_degree)
        model.load_state_dict(ckpt["state_dict"])
        return model
    def reset_params(self):
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=2e-4)
        nn.init.zeros_(self.net[-1].bias)
        
    def forward(self, x):
        return x[:,:-6] + self.net(x)