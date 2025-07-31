import sys, os, os.path as osp
import torch_geometric.nn.pool as pyg_pool
import numpy as np
import scipy
import torch
from torch import nn
import torch.nn.functional as F
import logging
import time
from matplotlib import pyplot as plt
from pytorch3d.transforms import (
    matrix_to_axis_angle,
    axis_angle_to_matrix,
    quaternion_to_matrix,
    matrix_to_quaternion,
)
from pytorch3d.ops import knn_points
import open3d as o3d
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gs_utils.gs_optim_helper import *
from mosca import MoSca, _compute_curve_topo_dist_, resample_curve, DQ_EPS
from featured_mosca import FeaturedMosca
from deformation_net import DeformationNet
import colorsys


# TODO: flexiblly swtich between GS mu parameterization, for later efficiency, have to directly save the location.

class FeaturedDynamicGS(nn.Module):
    def __init__(
            self,
            featured_scf: FeaturedMosca,
            deformation_net: DeformationNet,
            min_num_gs=32,
            max_scale=0.1,  # use sigmoid activation, can't be too large
            min_scale=0.001,
            max_sph_order=0,
            device=None,
        ):
        super().__init__()

        self.featured_scf = featured_scf
        self.scf = featured_scf
        self.deformation_net = deformation_net
        
        self.device = device if device is not None else torch.device("cpu")
        self.register_buffer("min_num_gs", torch.tensor(min_num_gs))
        self.register_buffer("max_scale", torch.tensor(max_scale))
        self.register_buffer("min_scale", torch.tensor(min_scale))
        self.register_buffer(
            "max_sph_order", torch.tensor(max_sph_order)
        )
        self.register_buffer("xyz", torch.empty(0, dtype=torch.float))
        
        # Changed to torch.nn.Parameter
        self.quat = nn.Parameter(torch.empty(0, dtype=torch.float))
        self.scale = nn.Parameter(torch.empty(0, dtype=torch.float))
        self.opacity = nn.Parameter(torch.empty(0, dtype=torch.float))
        self.sph = nn.Parameter(torch.empty(0, dtype=torch.float))

        self.register_buffer("attach_ind", torch.empty(0, dtype=torch.int32))
        self.register_buffer("ref_time", torch.empty(0, dtype=torch.int32))
        self._init_act(max_scale, min_scale)
        
        self.to(device)
    
    @classmethod
    def load_from_ckpt(cls, ckpt, featured_scf, deformation_net, device=None):
        min_num_gs = ckpt["min_num_gs"]
        max_scale = ckpt["max_scale"]
        min_scale = ckpt["min_scale"]
        max_sph_order = ckpt["max_sph_order"]
        xyz = ckpt["xyz"]
        quat = ckpt["quat"]
        scale = ckpt["scale"]
        opacity = ckpt["opacity"]
        sph = ckpt["sph"]
        attach_ind = ckpt["attach_ind"]
        ref_time = ckpt["ref_time"]
        f_d_gs = cls(featured_scf, deformation_net, min_num_gs, max_scale, min_scale, max_sph_order)
        f_d_gs.xyz.data = xyz
        
        # When loading from checkpoint, assign directly to the .data attribute of the Parameter
        f_d_gs.quat.data = quat
        f_d_gs.scale.data = scale
        f_d_gs.opacity.data = opacity
        f_d_gs.sph.data = sph

        f_d_gs.attach_ind.data = attach_ind
        f_d_gs.ref_time.data = ref_time
        f_d_gs.to(device)
        return f_d_gs

    def _init_act(self, max_s_value, min_s_value):
        def s_act(x):
            if isinstance(x, float):
                x = torch.tensor(x).squeeze()
            return min_s_value + torch.sigmoid(x) * (max_s_value - min_s_value)

        def s_inv_act(x):
            if isinstance(x, float):
                x = torch.tensor(x).squeeze()
            x = torch.clamp(
                x, min=min_s_value + 1e-6, max=max_s_value - 1e-6
            )  # ! clamp
            y = (x - min_s_value) / (max_s_value - min_s_value) + 1e-5
            y = torch.clamp(y, min=1e-5, max=1 - 1e-5)
            y = torch.logit(y)
            assert not torch.isnan(
                y
            ).any(), f"{x.min()}, {x.max()}, {y.min()}, {y.max()}"
            return y

        def o_act(x):
            if isinstance(x, float):
                x = torch.tensor(x).squeeze()
            return torch.sigmoid(x)

        def o_inv_act(x):
            if isinstance(x, float):
                x = torch.tensor(x).squeeze()
            return torch.logit(x)

        self.s_act = s_act
        self.s_inv_act = s_inv_act
        self.o_act = o_act
        self.o_inv_act = o_inv_act

    def append_new_gs(
        self,
        tid,
        xyz,
        quat,
        scale,
        opacity,
        rgb,
    ):
        _, attach_ind, _ = knn_points(xyz[None], self.featured_scf.node_xyz[tid][None], K=1)
        self.attach_ind = torch.cat([self.attach_ind, attach_ind[0, :, 0]], dim=0)
        self.ref_time = torch.cat([self.ref_time, torch.ones_like(attach_ind[0, :, 0]) * tid], dim=0)

        new_scale = self.s_inv_act(scale)
        new_opacity = self.o_inv_act(opacity)
        new_feature_dc = RGB2SH(rgb)
        new_feature_rest = torch.zeros(
            len(scale), 3 * (self.max_sph_order + 1) ** 2 - 3
        ).to(self.device)
        new_sph_feature = torch.cat([new_feature_dc, new_feature_rest], dim=1)
        
        self.xyz = torch.cat([self.xyz, xyz], dim=0)
        self.quat.data = torch.cat([self.quat.data, quat], dim=0)
        self.scale.data = torch.cat([self.scale.data, new_scale], dim=0)
        self.opacity.data = torch.cat([self.opacity.data, new_opacity], dim=0)
        self.sph.data = torch.cat([self.sph.data, new_sph_feature], dim=0)

    def forward(self, tid, active_sph_order=None, nn_fusion=None):
        gs_feature = self.featured_scf.interpolate_feature(
            attach_node_id=self.attach_ind,
            query_xyz=self.xyz,
            query_tid=self.ref_time
        )
        xyz, _, _, _, _ = self.deformation_net(
            gs_feature, tid
        )
        if active_sph_order is None:
            active_sph_order = self.max_sph_order
        sph = self.sph[:, :3 * sph_order2nfeat(active_sph_order)]

        rotation = quaternion_to_matrix(self.quat)
        scale = self.s_act(self.scale)
        opacity = self.o_act(self.opacity)

        return xyz, rotation, scale, opacity, sph
    
    def get_node_sinning_w_acc(self, reduce="sum"):
        sk_ind, sk_w, _ = self.featured_scf.get_skinning_weights(
            query_xyz=self.get_xyz(),
            query_t=self.ref_time,
            attach_ind=self.attach_ind
        )
        sk_ind, sk_w = sk_ind.reshape(-1), sk_w.reshape(-1)
        acc_w = torch.zeros_like(self.scf.node_xyz[0, :, 0])
        acc_w = acc_w.scatter_reduce(0, sk_ind, sk_w, reduce=reduce, include_self=False)
        return acc_w
    
    def get_xyz(self):
        return self.xyz
    
    @property
    def N(self):
        return self.xyz.shape[0]

def RGB2SH(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

def sph_order2nfeat(order):
    return (order + 1) ** 2