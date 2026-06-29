import os, sys, os.path as osp
import torch
import torch.nn.functional as F

sys.path.append(osp.dirname(osp.abspath(__file__)))
# sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), ".."))

# get from env variable
try:
    GS_BACKEND = os.environ["GS_BACKEND"]
except:
    # GS_BACKEND = "native"
    print(f"No GS_BACKEND env var specified, for now use gof backend")
    GS_BACKEND = "gof"
GS_BACKEND = GS_BACKEND.lower()
print(f"GS_BACKEND: {GS_BACKEND.lower()}")

if GS_BACKEND == "native":
    from gauspl_renderer_native import render_cam_pcl
elif GS_BACKEND == "gof":
    from gauspl_renderer_gof import render_cam_pcl
elif GS_BACKEND == "native_add3":
    from gauspl_renderer_native_add3 import render_cam_pcl
else:
    raise ValueError(f"Unknown GS_BACKEND: {GS_BACKEND.lower()}")

from sh_utils import RGB2SH, SH2RGB
from lib_mosca.util import get_normal, reflect

def render(
    gs_param,
    H,
    W,
    K,
    T_cw,
    bg_color=[1.0, 1.0, 1.0],
    scale_factor=1.0,
    opa_replace=None,
    bg_cache_dict=None,
    colors_precomp=None,
    add_buffer=None,

    # pbr related
    s_model=None,
    d_model=None,
    deform_env=None,
    brdf=None,
    view_pos=None,

    detach_diffuse=False,
    view_ind=None,
):
    normal = None

    # * Core render interface
    # prepare gs5 param in world system

    if torch.is_tensor(gs_param[0]):  # direct 5 tuple
        mu, fr, s, o, sph = gs_param
    else:
        mu, fr, s, o, sph = gs_param[0]
        for i in range(1, len(gs_param)):
            mu = torch.cat([mu, gs_param[i][0]], 0)
            fr = torch.cat([fr, gs_param[i][1]], 0)
            s = torch.cat([s, gs_param[i][2]], 0)
            o = torch.cat([o, gs_param[i][3]], 0)
            sph = torch.cat([sph, gs_param[i][4]], 0)
    if opa_replace is not None:
        assert isinstance(opa_replace, float)
        o = torch.ones_like(o) * opa_replace
    s = s * scale_factor

    # cvt to cam frame
    assert T_cw.ndim == 2 and T_cw.shape[1] == T_cw.shape[0] == 4
    R_cw, t_cw = T_cw[:3, :3], T_cw[:3, 3]
    mu_cam = torch.einsum("ij,nj->ni", R_cw, mu) + t_cw[None]
    fr_cam = torch.einsum("ij,njk->nik", R_cw, fr)

    if colors_precomp is None:
        if brdf is not None:
            assert s_model is not None and d_model is not None, "s_model and d_model must be provided when brdf is not None"
            assert view_pos is not None, "view_pos must be provided when brdf is not None"

            position = mu
            view_dir = torch.nn.functional.normalize(view_pos - position, dim=-1)
            scale = s
            rotation = fr
            residual_normal = torch.cat([s_model.get_residual_normal, d_model.get_residual_normal], 0)
            normal = get_normal(scale, rotation, view_dir, residual_normal)

            d_reflvec = 0.0
            d_reflvec = torch.tensor(d_reflvec)
            d_reflvec = d_reflvec.to(s_model.device)
            
            if brdf is not None:
                brdf.build_mips()
            if deform_env is not None and view_ind is not None and False:
                reflvec = F.normalize(reflect(view_dir, normal), dim=-1)
                d_reflvec = deform_env.step(reflvec.detach(), view_ind)

            diffuse = torch.cat([s_model.get_diffuse, d_model.get_diffuse], 0)
            specular = torch.cat([s_model.get_specular, d_model.get_specular], 0)
            roughness = torch.cat([s_model.get_roughness, d_model.get_roughness], 0)
            
            if detach_diffuse:
                diffuse = diffuse.detach()

            color = brdf.shade(
                mu[None, None, ...].detach(),
                normal[None, None, ...],
                d_reflvec[None, None, ...],
                diffuse[None, None, ...],
                specular[None, None, ...],
                roughness[None, None, ...],
                view_pos[None, None, ...],
            )
            colors_precomp = color.squeeze()
            colors_precomp = torch.clamp(colors_precomp, 0.0, 1.0)

    # convert normal to cam frame, for shading, note that the normal is in world frame, so we need to rotate it to cam frame, and the normal is in world frame, so we only need to rotate it, no need to translate
    normal_cam_buffer = torch.einsum("ij, nj->ni", R_cw, normal) if normal is not None else None

    # render
    render_dict = render_cam_pcl(
        mu_cam,
        fr_cam,
        s,
        o,
        sph,
        H,
        W,
        CAM_K=K,
        bg_color=bg_color,
        colors_precomp=colors_precomp,
        add_buffer=add_buffer,
        normal_buffer=normal_cam_buffer,
    )
    if bg_cache_dict is not None:
        render_dict = fast_bg_compose_render(bg_cache_dict, render_dict, bg_color)
    return render_dict


def fast_bg_compose_render(bg_cache_dict, render_dict, bg_color=[1.0, 1.0, 1.0]):
    assert GS_BACKEND == "native", "GOF does not support this now"

    # manually compose the fg
    # ! warning, be careful when use the visibility masks .etc, watch the len
    fg_rgb, bg_rgb = render_dict["rgb"], bg_cache_dict["rgb"]
    fg_alpha, bg_alpha = render_dict["alpha"], bg_cache_dict["alpha"]
    fg_dep, bg_dep = render_dict["dep"], bg_cache_dict["dep"]
    _fg_alp = torch.clamp(fg_alpha, 1e-8, 1.0)
    _bg_alp = torch.clamp(bg_alpha, 1e-8, 1.0)
    fg_dep_corr = fg_dep / _fg_alp
    bg_dep_corr = bg_dep / _bg_alp
    fg_in_front = (fg_dep_corr < bg_dep_corr).float()
    # compose alpha
    alpha_fg_front_compose = fg_alpha + (1.0 - fg_alpha) * bg_alpha
    alpha_fg_behind_compose = bg_alpha + (1.0 - bg_alpha) * fg_alpha
    alpha_composed = alpha_fg_front_compose * fg_in_front + alpha_fg_behind_compose * (
        1.0 - fg_in_front
)
    alpha_composed = torch.clamp(alpha_composed, 0.0, 1.0)
    # compose rgb
    bg_color = torch.as_tensor(bg_color, device=fg_rgb.device, dtype=fg_rgb.dtype)
    rgb_fg_front_compose = (
        fg_rgb * fg_alpha
        + bg_rgb * (1.0 - fg_alpha) * bg_alpha
        + (1.0 - fg_alpha) * (1.0 - bg_alpha) * bg_color[:, None, None]
    )
    rgb_fg_behind_compose = (
        bg_rgb * bg_alpha
        + fg_rgb * (1.0 - bg_alpha) * fg_alpha
        + (1.0 - bg_alpha) * (1.0 - fg_alpha) * bg_color[:, None, None]
    )
    rgb_composed = rgb_fg_front_compose * fg_in_front + rgb_fg_behind_compose * (
        1.0 - fg_in_front
    )
    # compose dep
    dep_fg_front_compose = (
        fg_dep_corr * fg_alpha + bg_dep_corr * (1.0 - fg_alpha) * bg_alpha
    )
    dep_fg_behind_compose = (
        bg_dep_corr * bg_alpha + fg_dep_corr * (1.0 - bg_alpha) * fg_alpha
    )
    dep_composed = dep_fg_front_compose * fg_in_front + dep_fg_behind_compose * (
        1.0 - fg_in_front
    )
    return {
        "rgb": rgb_composed,
        "dep": dep_composed,
        "alpha": alpha_composed,
        "visibility_filter": render_dict["visibility_filter"],
        "viewspace_points": render_dict["viewspace_points"],
        "radii": render_dict["radii"],
        "dyn_rgb": fg_rgb,
        "dyn_dep": fg_dep,
        "dyn_alpha": fg_alpha,
    }
