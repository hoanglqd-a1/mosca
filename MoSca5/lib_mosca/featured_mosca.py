import torch
import torch.nn as nn
import torch.nn.functional as F
import logging, time
import sys, os, os.path as osp
from pytorch3d.ops import knn_points
from tqdm import tqdm
from matplotlib import pyplot as plt

from mosca import MoSca
from deformation_net import DeformationNet

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class FeaturedMosca(nn.Module):
    node_xyz: torch.Tensor
    def __init__(self, 
            scf: MoSca, 
            feature_dim: int = 64,
        ):
        super().__init__()
        if scf is not None:
            self.node_feature = nn.Parameter(
                torch.zeros(scf.M, feature_dim)
            )
            self.register_buffer("node_xyz", scf.node_xyz.data.clone())
            self.register_buffer("t_list", scf._t_list.data.clone())
            self.register_buffer("feature_dim", torch.tensor(feature_dim))
            self.register_buffer("topo_knn_ind", scf.topo_knn_ind)
            self.register_buffer("topo_knn_mask", scf.topo_knn_mask)
            self.register_buffer("node_sigma", scf.node_sigma)
            self.register_buffer("skinning_k", scf.skinning_k)
            self.register_buffer("spatial_unit", scf.spatial_unit)
    
    @classmethod
    def load_from_ckpt(cls, ckpt):
        node_xyz = ckpt["node_xyz"]
        t_list = ckpt["t_list"]
        feature_dim = int(ckpt["feature_dim"].item())
        topo_knn_ind = ckpt["topo_knn_ind"]
        topo_knn_mask = ckpt["topo_knn_mask"]
        node_sigma = ckpt["node_sigma"]
        node_feature = ckpt["node_feature"]
        skinning_k = ckpt["skinning_k"]
        spatial_unit = ckpt["spatial_unit"]
        feature_mosca = cls(None, feature_dim)
        feature_mosca.register_buffer("node_xyz", node_xyz)
        feature_mosca.register_buffer("t_list", t_list)
        feature_mosca.register_buffer("feature_dim", torch.tensor(feature_dim))
        feature_mosca.register_buffer("topo_knn_ind", topo_knn_ind)
        feature_mosca.register_buffer("topo_knn_mask", topo_knn_mask)
        feature_mosca.register_buffer("node_sigma", node_sigma)
        feature_mosca.register_buffer("skinning_k", skinning_k)
        feature_mosca.register_buffer("spatial_unit", spatial_unit)
        feature_mosca.node_feature = nn.Parameter(node_feature)
        return feature_mosca

    def update_node_xyz(self, deformation_net: DeformationNet):
        with torch.no_grad():
            self.node_xyz.data, _, _, _, _ = deformation_net(self.node_feature)

    def interpolate_feature(
        self,
        attach_node_id,
        query_xyz,
        query_tid,
    ):
        if isinstance(query_tid, int) or query_tid.ndim == 0:
            query_tid = torch.ones_like(query_xyz[:, 0]).long() * query_tid
        N = len(query_tid)
        assert len(query_xyz) == N and query_xyz.shape == (N, 3)
        sk_ind, sk_w, sk_w_sum = self.get_skinning_weights(
            query_xyz=query_xyz,
            query_t=query_tid,
            attach_ind=attach_node_id,
        )

        neighbor_features = self.node_feature[sk_ind]
        weighted_features = neighbor_features * sk_w.unsqueeze(-1)
        interpolated_features = weighted_features.sum(dim=1)
        return interpolated_features
        
    def get_skinning_weights(
        self,
        query_xyz,
        query_t,
        attach_ind
    ):
        sk_ind = self.topo_knn_ind[attach_ind]
        sk_mask = self.topo_knn_mask[attach_ind]
        
        sk_ref_node_xyz = self.get_async_knns(query_t, sk_ind)
        sq_dist_to_sk_node = (query_xyz[:, None, :] - sk_ref_node_xyz) ** 2
        sq_dist_to_sk_node = sq_dist_to_sk_node.sum(-1)
        sk_w_un = (
            torch.exp(
                -sq_dist_to_sk_node / (2 * (self.node_sigma.squeeze(-1) ** 2)[sk_ind])
            ) + 1e-6
        )
        sk_w_sum = sk_w_un.sum(-1)
        sk_w = sk_w_un / torch.clamp(sk_w_sum, min=1e-6)[:, None]
        return sk_ind, sk_w, sk_w_sum
        
    def get_async_knns(self, t, knn_ind):
        assert t.ndim == 1 and knn_ind.ndim == 2 and len(t) == len(knn_ind)
        # self._node_XXXX[t,knn_ind]
        with torch.no_grad():
            flat_sk_ind = t[:, None] * self.M + knn_ind
        sk_ref_node_xyz = self.node_xyz.reshape(-1, 3)[flat_sk_ind, :]  # N,K,3
        return sk_ref_node_xyz
    
    @property
    def M(self):
        return self.node_xyz.shape[1]