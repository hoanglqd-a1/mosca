import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class DeformationNet(nn.Module):
    def __init__(self, 
            feature_dim: int = 32, 
            spherical_harmonics_degree: int = 3
            ):
        super().__init__()
        self.register_buffer("feature_dim", torch.tensor(feature_dim))
        self.register_buffer("spherical_harmonics_degree", torch.tensor(spherical_harmonics_degree))
        input_dim = feature_dim + 1 # feature + queried time
        output_dim = 8 + 3 * (spherical_harmonics_degree + 1) ** 2 # output include quaternion, scale, opacity, spherical harmonic
        self.seq1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
        )
        self.seq2 = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

        self._init_last_layers_small()

    def _init_last_layers_small(self):
        # Manually access the last layers
        last1 = self.seq1[-1]
        last2 = self.seq2[-1]

        # Small normal init for weights
        init.normal_(last1.weight, mean=0.0, std=1e-2)
        init.constant_(last1.bias, 0.0)

        init.normal_(last2.weight, mean=0.0, std=1e-2)
        init.constant_(last2.bias, 0.0)

    @classmethod
    def load_from_ckpt(cls, ckpt):
        # Get original configuration
        feature_dim = int(ckpt["feature_dim"].item())
        sh_degree = int(ckpt["spherical_harmonics_degree"].item())

        # Instantiate a new model with the same config
        model = cls(feature_dim=feature_dim, spherical_harmonics_degree=sh_degree)

        # Load parameters (assume ckpt["state_dict"] holds the weights)
        model.load_state_dict(ckpt, strict=True)

        return model
        
    def forward(self, gaussian_feature, queried_time, flag=True):
        if isinstance(queried_time, (int, float)):
            queried_time = torch.full(
                (gaussian_feature.shape[0], 1),
                queried_time,
                device=gaussian_feature.device,
                dtype=gaussian_feature.dtype
            )
        elif isinstance(queried_time, torch.Tensor):
            if queried_time.ndim == 0:
                queried_time = queried_time.expand(gaussian_feature.shape[0]).unsqueeze(1)
            elif queried_time.ndim == 1:
                queried_time = queried_time.unsqueeze(1)
        x = torch.cat([gaussian_feature, queried_time], dim=1)
        xyz = self.seq1(x)
        if not flag:
            return xyz, None, None, None, None
        x = self.seq2(x)

        quat = x[:, 0:4]
        scale = x[:, 4:7]
        opacity = x[:, 7:8]
        sph = x[:, 8:]
        # return xyz, quat, scale, opacity, sph
        return xyz, quat, scale, opacity, sph
