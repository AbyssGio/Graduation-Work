import numpy as np
import torch
import cv2
import einops

from utils.rgbd2pcd import compute_xyz, get_xyz, get_xyz_cpp
import torch.nn.functional as F


def inverse_warp_numpy(ref_depth, cur_data, R, t, camera):
    # 寻找在当前 cur_data 中对应的 ref_depth 的点的插值出的深度值
    _, h, w = ref_depth.shape
    pt_current = compute_xyz(ref_depth, camera).squeeze().transpose(2, 0, 1).reshape(3, -1)
    XYZ = np.matmul(R, pt_current)
    X = (XYZ[0, :] + t[0]).reshape(h, w)
    Y = (XYZ[1, :] + t[1]).reshape(h, w)
    Z = (XYZ[2, :] + t[2]).reshape(h, w)

    u = camera["fx"] * X / Z + camera["cx"]
    v = camera["fy"] * Y / Z + camera["cy"]
    u = u.astype(np.float32)
    v = v.astype(np.float32)
    u[u < 0] = 0
    v[v < 0] = 0
    reprojected_data = cv2.remap(cur_data, u, v, cv2.INTER_NEAREST)
    # uv_mask = ((u >= 0) + (u < w) + (v >= 0) + (v < h))
    # uv = np.stack([u, v], dim=2).view(h, w, 2)
    return reprojected_data


def w2c(X, Y, Z, fx, fy, cx, cy, h, w):
    ffx = einops.repeat(fx, 'bs -> bs 1 h w', h=h, w=w)
    ffy = einops.repeat(fy, 'bs -> bs 1 h w', h=h, w=w)
    ccx = einops.repeat(cx, 'bs -> bs 1 h w', h=h, w=w)
    ccy = einops.repeat(cy, 'bs -> bs 1 h w', h=h, w=w)

    U_proj = ffx * X / Z + ccx  # horizontal pixel coordinate
    V_proj = ffy * Y / Z + ccy  # vertical pixel coordinate
    return U_proj, V_proj


def inverse_warp_cpp(ref_depth, cur_data, R, t, fx, fy, cx, cy):
    b, _, h, w = ref_depth.size()
    pt = get_xyz_cpp(ref_depth, fx, fy, cx, cy)
    XYZ_ = torch.bmm(R, pt.view(b, 3, -1))
    X = (XYZ_[:, 0, :] + t[:, 0].unsqueeze(1)).view(-1, 1, h, w)
    Y = (XYZ_[:, 1, :] + t[:, 1].unsqueeze(1)).view(-1, 1, h, w)
    Z = (XYZ_[:, 2, :] + t[:, 2].unsqueeze(1)).view(-1, 1, h, w)
    U_proj = fx.unsqueeze(1) * X / Z + cx.unsqueeze(1)
    V_proj = fy.unsqueeze(1) * Y / Z + cy.unsqueeze(1)
    U_proj_normalized = (2 * U_proj / (w - 1) - 1).view(b, -1)
    V_proj_normalized = (2 * V_proj / (h - 1) - 1).view(b, -1)
    pixel_coords = torch.stack([U_proj_normalized, V_proj_normalized], dim=2).view(b, h, w, 2)  # [B, H, W, 2]
    reprojected_data = F.grid_sample(cur_data, pixel_coords, align_corners=True)
    return reprojected_data

def inverse_warp_pytorch(ref_depth, cur_data, R, t, fx, fy, cx, cy, padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.
    """
    b, _, h, w = ref_depth.size()
    pt = get_xyz(ref_depth, fx, fy, cx, cy)
    XYZ_ = torch.bmm(R, pt.view(b, 3, -1))
    X = (XYZ_[:, 0, :] + t[:, 0].unsqueeze(1)).view(-1, 1, h, w)
    Y = (XYZ_[:, 1, :] + t[:, 1].unsqueeze(1)).view(-1, 1, h, w)
    Z = (XYZ_[:, 2, :] + t[:, 2].unsqueeze(1)).view(-1, 1, h, w)
    # compute pixel coordinates
    U_proj, V_proj = w2c(X, Y, Z, fx, fy, cx, cy, h, w)
    # normalization to [-1, 1], required by torch.nn.functional.grid_sample
    U_proj_normalized = (2 * U_proj / (w - 1) - 1).view(b, -1)
    V_proj_normalized = (2 * V_proj / (h - 1) - 1).view(b, -1)
    if padding_mode == 'zeros':
        U_proj_mask = ((U_proj_normalized > 1) + (U_proj_normalized < -1)).detach()
        U_proj_normalized[U_proj_mask] = 2
        V_proj_mask = ((V_proj_normalized > 1) + (V_proj_normalized < -1)).detach()
        V_proj_normalized[V_proj_mask] = 2

    pixel_coords = torch.stack([U_proj_normalized, V_proj_normalized], dim=2).view(b, h, w, 2)  # [B, H, W, 2]
    reprojected_data = F.grid_sample(cur_data, pixel_coords, align_corners=True)
    return reprojected_data


