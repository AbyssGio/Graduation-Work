import cv2
import torch
import os
import numpy as np
import torch.nn.functional as F

from random import randint
from torch.utils.data import DataLoader

from cfg.datasets.BlenderGlass_dataset import BlenderGlass
from train.metrics import Metrics, surface_normal_cos_loss, surface_normal_l1_loss, depth_loss, pairwise_L1_depth_loss, \
    ConsLoss
from cfg.datasets.cleargrasp_dataset import ClearGrasp
from utils.normal2depth import normal_to_depth
from utils.rgbd2pcd import get_surface_normal_from_xyz, get_xyz
from utils.visualize import vis_surface_normal_3d, vis_pt_3d, vis_mask, vis_depth

delta_list = [1.04, 1.09, 1.14]
img_h = img_w = 256
# dataset = BlenderGlass("/home/ctwo/glass", "test", "scene1", img_h, img_w, max_norm=True)
dataset = ClearGrasp("/home/ctwo/cleargrasp_dataset", "val", img_h, img_w, max_norm=True)
batch_data = dataset[0]

data_dict = {
    "gt_depth": batch_data["gt_depth"].unsqueeze(0),
    "camera": {
        "fx": batch_data["fx"].unsqueeze(0),
        "fy": batch_data["fy"].unsqueeze(0),
        "cx": batch_data["cx"].unsqueeze(0),
        "cy": batch_data["cy"].unsqueeze(0),
    },
    "depth_gt_sn": batch_data["depth_gt_sn"].unsqueeze(0),
    "mask": batch_data["mask"].unsqueeze(0),
    "sn_mask": batch_data["sn_mask"].unsqueeze(0)
}

gt_depth = data_dict["gt_depth"]
gt_depth = gt_depth / torch.amax(gt_depth)

metric = Metrics()
cons_loss = ConsLoss(reduction="mean")
for delta in delta_list:
    noise_min = -(delta - 1) * gt_depth
    noise_max = (delta - 1) * gt_depth
    gt_depth_noise = torch.rand(1, 1, img_h, img_w)
    gt_depth_noise = gt_depth_noise * (noise_max - noise_min) + noise_min
    predict_depth = gt_depth + gt_depth_noise

    mask = data_dict["mask"] > 0
    _mask = mask.squeeze(1)
    sn_mask = data_dict["sn_mask"] > 0
    sn_mask = sn_mask.squeeze(1)
    camera = data_dict["camera"]
    gt_xyz = get_xyz(gt_depth, camera["fx"], camera["fy"], camera["cx"], camera["cy"])
    predict_xyz = get_xyz(predict_depth, camera["fx"], camera["fy"], camera["cx"], camera["cy"])
    # 1-point sn
    gt_sn = get_surface_normal_from_xyz(gt_xyz)
    p_loss = surface_normal_l1_loss(predict_depth, gt_sn, mask, camera, reduction="mean", beta=0.5)
    # kdtree sn
    # gt_xyz = gt_xyz.permute(0, 2, 3, 1)
    kdtree_gt_sn = get_surface_normal_from_xyz(gt_xyz, method="kdtree")
    # gt_pt = gt_xyz[_mask > 0]
    # gt_pt = gt_pt.permute(1, 0)
    # vis_pt_3d(gt_pt.numpy(), v_size=None)
    # vis_surface_normal_3d(gt_pt.numpy(), v_size=0.005)
    kdtree_predict_sn = get_surface_normal_from_xyz(predict_xyz, method="kdtree", max_nn=50)
    # consistency_loss = cons_loss(predict_depth, kdtree_predict_sn, data_dict["mask"], gt_depth, camera)
    # kdtree_loss = 1 - F.cosine_similarity(kdtree_predict_sn, kdtree_gt_sn, dim=1)
    # kdtree_loss = F.smooth_l1_loss(kdtree_predict_sn*sn_mask, gt_sn*sn_mask, reduction="sum") / torch.sum(sn_mask)
    # vis_mask(sn_mask.squeeze())
    # vis_mask(_mask.squeeze())
    # kdtree_loss = torch.mean(kdtree_loss[sn_mask > 0])

    d = normal_to_depth(gt_xyz, gt_sn, camera)


    # depth_failing_105 = metric.depth_failing_105(predict_depth, gt_depth, mask)
    # depth_failing_110 = metric.depth_failing_110(predict_depth, gt_depth, mask)
    # depth_failing_115 = metric.depth_failing_115(predict_depth, gt_depth, mask)
    rmse = metric.masked_depth_rmse(d, gt_depth, mask)
    mae = metric.masked_depth_mae(d, gt_depth, mask)
    # rel = metric.masked_depth_rel(d, gt_depth, mask)
    diff = (gt_depth-d).abs().squeeze().numpy()
    vis_depth(diff*1000, visualize=True, mi=0, ma=1, color_map="gray")
    # pairwise_l1 = pairwise_L1_depth_loss(predict_depth, gt_depth, mask)
    # print("delta_{} result:".format(delta))
    # print("depth failing 1.05, 1.10, 1.15: {:.2f}%, {:.2f}%, {:.2f}%".format(
    #     depth_failing_105.item()*100, depth_failing_110.item()*100, depth_failing_115.item()*100
    # ))
    print("rmse, mae, rel: {:.4f}, {:.4f}, {:.4f}".format(rmse, mae, 0))
    # print("1-point sn loss: {:.4f}".format(p_loss.item()))
    # print("depth_loss: {:.4f}".format(depth_loss(predict_depth, gt_depth, mask, beta=0.02, reduction="mean")))
    # print("pair-wise L1 loss: {:.4f}".format(pairwise_l1.item()))
    # print("consistency loss: {:.4f}".format(consistency_loss.item()))
    # print("kdtree sn loss: {:.4f}".format(kdtree_loss.cpu().item()))
    break
