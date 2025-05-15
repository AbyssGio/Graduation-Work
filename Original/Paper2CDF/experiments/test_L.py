import random
from time import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from model.df_net import DFNet
from utils.rgbd2pcd import random_xyz_sampling, get_xyz, compute_xyz
from utils.visualize import vis_depth, vis_surface_normal, vis_surface_normal_3d, vis_pt_3d
from skimage.transform import resize
from torchvision import transforms

from cfg.datasets.BlenderGlass_dataset import png_depth_loader, safe_resize, BlenderGlass
from cfg.datasets.cleargrasp_dataset import mask_loader
from cfg.datasets.RealTime_dataset import RealTimeCaptureDataset
import matplotlib.pyplot as plt
from utils import config_loader, rgbd2pcd
from utils import exr_handler

import os

#
# np.random.seed(0)
# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
# torch.cuda.manual_seed_all(0)
# random.seed(0)

test_scene = "seq1"
ds_root = "/home/ctwo/blender_glass_data"
ds_path = os.path.join(ds_root, test_scene)
out_path = os.path.join(ds_path, "depth_refine1")
vis_path = os.path.join(ds_path, "vis_depth")
vis_surface_normal_path = os.path.join(ds_path, "vis_sn")

outputImgHeight = 480
outputImgWidth = 640

cfg_file_path = "../experiments/dfnet_pt_L50.yml"
# cfg_file_path = "../experiments/swin_pt_spade_config.yml"
cfg = config_loader.CFG.read_from_yaml(cfg_file_path)
device = torch.device("cuda")
# L_list = [10, 20, 30, 40, 50, 60]
L_list = [50]
start_time = time()
depth_factor = 2000
dataset = BlenderGlass(root=ds_root,
                       scene=test_scene,
                       img_h=cfg.general.frame_h,
                       img_w=cfg.general.frame_w,
                       rgb_aug=False,
                       max_norm=True,
                       depth_factor=depth_factor
                       )

dataloader = DataLoader(
    dataset,
    shuffle=False,
    num_workers=8,
    batch_size=1,
)
print("Total frame: {}".format(len(dataset)))

for depth_L in L_list:
    cfg.model.depth_plane_num = depth_L
    model = DFNet(cfg)
    snapshot = torch.load(cfg.general.pretrained_weight)
    model.load_state_dict(snapshot["model"])
    model = model.to(device)
    for idx, batch in enumerate(dataloader):
        png_name = dataset.img_name_list[idx]
        depth_name = dataset.depth_list[idx]
        mask_name = dataset.mask_list[idx]
        original_mask = mask_loader(mask_name)
        original_depth = png_depth_loader(depth_name)
        original_depth_ = png_depth_loader(depth_name)
        # batch["R_mat"] = torch.ones_like(batch["R_mat"]).float()
        # batch["t_vec"] = torch.zeros_like(batch["t_vec"]).float()
        # batch["forward_rgb"] = batch["cur_rgb"].clone()
        # batch["forward_depth"] = batch["cur_gt_depth"].clone()
        # batch["forward_color"] = batch["cur_color"].clone()

        for k in batch:
            batch[k] = batch[k].to(device)
        model.eval()
        eval_time_start = time()
        with torch.no_grad():
            output_depth = model(batch)
            output_depth = output_depth.squeeze().cpu().numpy()
        output_depth = safe_resize(output_depth, outputImgWidth, outputImgHeight) * depth_factor
        original_depth[original_mask > 0] = output_depth[original_mask > 0]
        print("eval time used: {}".format(time() - eval_time_start))
        png_filename = "{}.png".format(png_name)
        # vis_save_path = os.path.join(vis_path, png_filename)
        # vis_sn_save_path = os.path.join(vis_surface_normal_path, png_filename)
        # vis_depth(original_depth / np.amax(original_depth), 0, 1, visualize=False,
        #           save_path=None, color_map='Greys')
        # vis_surface_normal(original_depth, dataset.camera_origin, visualize=True,
        #                    save_path=None, color_map="RdBu_r")
        # vis_depth(np.abs(original_depth_ - original_depth), 0, 1, color_map="gray", visualize=False,
        #           save_path="./diff_L{}.png".format(depth_L))
        # out_file_path = os.path.join(out_path, png_filename)
        # exr_handler.png_writer(original_depth, out_file_path)
        print("{} predict complete".format(png_name))
        break
print("test end, time used: {:.2f}s".format(time() - start_time))
