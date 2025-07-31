import torch
import torch.nn as nn

class Deformation(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.seq1 = nn.Sequential(
            nn.Linear(feature_dim + 4, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )
        self.register_buffer("feature_dim", torch.tensor(feature_dim))
    def forward(self, feature, tid, xyz_cannon):
        if isinstance(tid, int):
            tid = torch.ones(feature.shape[0], 1, device=feature.device) * tid
        x = torch.cat([feature, tid, xyz_cannon], dim=1)
        x = self.seq1(x)
        delta_xyz = x
        xyz = xyz_cannon + delta_xyz
        return xyz
    
    @classmethod
    def load_from_ckpt(cls, ckpt):
        feature_dim = int(ckpt["feature_dim"].item())
        model = cls(feature_dim=feature_dim)
        model.load_state_dict(ckpt, strict=True)
        return model