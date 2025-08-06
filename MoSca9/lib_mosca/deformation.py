import torch
import torch.nn as nn

class Deformation(nn.Module):
    def __init__(self, xyz_encoding_degree=10, time_encoding_degree=4):
        super().__init__()
        feature_dim = 3 * (2 * xyz_encoding_degree + 1)
        time_dim = 2 * time_encoding_degree + 1
        self.seq1 = nn.Sequential(
            nn.Linear(feature_dim + time_dim, 256),
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
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )
        self.register_buffer("xyz_encoding_degree", torch.tensor(xyz_encoding_degree, dtype=torch.int))
        self.register_buffer("time_encoding_degree", torch.tensor(time_encoding_degree, dtype=torch.int))
    def forward(self, query_xyz, tid, all_view_flag=False, **kwargs):
        node_cnt = query_xyz.shape[0]
        if isinstance(tid, int):
            tid = torch.ones(node_cnt, 1, device=query_xyz.device) * tid
        encoded_xyz = self.positional_encoding(query_xyz, self.xyz_encoding_degree)
        encoded_time = self.positional_encoding(tid, self.time_encoding_degree)
        if all_view_flag:
            num_time_steps = kwargs["num_time_steps"]
            encoded_xyz = encoded_xyz.repeat(num_time_steps, 1)
            query_xyz = query_xyz.repeat(num_time_steps, 1)
            
        x = torch.cat([encoded_xyz, encoded_time], dim=1)
        x = self.seq1(x)
        delta_xyz = x
        xyz = query_xyz + delta_xyz

        return xyz
    
    @classmethod
    def load_from_ckpt(cls, ckpt):
        xyz_encoding_degree = ckpt["xyz_encoding_degree"]
        time_encoding_degree = ckpt["time_encoding_degree"]
        model = cls(xyz_encoding_degree, time_encoding_degree)
        model.load_state_dict(ckpt, strict=True)
        return model
    
    def positional_encoding(self, feature, encoding_degree):
        powers = 2.**torch.arange(encoding_degree).to(feature.device)
        freq_bands = powers * torch.pi

        scaled_feature = feature.unsqueeze(-1) * freq_bands
        sin_features = torch.sin(scaled_feature)
        cos_features = torch.cos(scaled_feature)
        sin_features_flatten = sin_features.flatten(start_dim=1)
        cos_features_flatten = cos_features.flatten(start_dim=1)

        encoded_feature = torch.cat([feature, sin_features_flatten, cos_features_flatten], dim=1)
        return encoded_feature