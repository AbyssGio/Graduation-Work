import numpy as np
import torch
import cv2

from utils.inverse_warp import w2c
from utils.rgbd2pcd import compute_xyz, get_xyz
import torch.nn.functional as F


def mean_on_mask(diff, mask):
    # c = diff.size()[1]
    # mask = valid_mask.repeat(1, c, 1, 1)
    if mask.sum() > 0:
        mean_value = (diff * mask).sum() / mask.sum()
    else:
        mean_value = 0
    return mean_value


def photometric_geometry_loss(cur_img, forward_img, 
                              cur_depth, forward_depth, 
                              R, t, camera,
                              mask,
                              ssim_fn=None,
                              weight=(1.0, 0.15)):
    
    b, _, h, w = cur_depth.size()
    pt = get_xyz(cur_depth, camera["fx"], camera["fy"], camera["cx"], camera["cy"]).float()
    XYZ_ = torch.bmm(R, pt.view(b, 3, -1))
    X = (XYZ_[:, 0, :] + t[:, 0].unsqueeze(1)).view(-1, 1, h, w)
    Y = (XYZ_[:, 1, :] + t[:, 1].unsqueeze(1)).view(-1, 1, h, w)
    Z = (XYZ_[:, 2, :] + t[:, 2].unsqueeze(1)).view(-1, 1, h, w)
    U_proj, V_proj = w2c(X, Y, Z, camera["fx"], camera["fy"], camera["cx"], camera["cy"], h, w)
    U_proj_normalized = (2 * U_proj / (w - 1) - 1).view(b, -1)
    V_proj_normalized = (2 * V_proj / (h - 1) - 1).view(b, -1)
    pixel_coords = torch.stack([U_proj_normalized, V_proj_normalized], dim=2).view(b, h, w, 2)  # [B, H, W, 2]

    reprojected_mask = torch.abs(pixel_coords) < 1
    pixel_coords[~reprojected_mask] = 2
    reprojected_color = F.grid_sample(forward_img, pixel_coords, align_corners=True)

    reprojected_sample_depth = F.grid_sample(forward_depth, pixel_coords, align_corners=True)

    diff_depth = (Z - reprojected_sample_depth).abs() / (Z + reprojected_sample_depth)
    loss_mask = ((torch.sum(reprojected_color, 1) > 0).float() * torch.squeeze(mask).float()).unsqueeze(1)
    
    photometric_loss = (cur_img - reprojected_color).abs().mean(dim=1, keepdim=True)
    # 无监督常用 SSIM, 有监督直接用 gt 就好了
    if ssim_fn is not None:
        ssim_map = ssim_fn(cur_img, reprojected_color)
        photometric_loss = (0.15 * photometric_loss + 0.85 * ssim_map)
    photometric_loss = mean_on_mask(photometric_loss, loss_mask)
    geometry_loss = mean_on_mask(diff_depth, loss_mask)
    return photometric_loss*weight[0] + geometry_loss*weight[1]


def photometric_geometry_loss_v2(pred, forward_window_size, batch, camera):
    out = 0.0
    for w in range(1, forward_window_size+1):
        out = out + photometric_geometry_loss(
            batch["cur_rgb"], batch["forward_{}_rgb".format(w)], pred, batch["forward_{}_depth".format(w)],
            batch["forward_{}_R_mat".format(w)], batch["forward_{}_t_vec".format(w)],
            camera, batch["mask"], ssim_fn=None
        )
    return out
