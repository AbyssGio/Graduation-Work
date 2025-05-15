import os
import numpy as np
import torch

from PIL import Image
from scipy.spatial.transform import Rotation
from torchvision import transforms
from torch.utils.data import Dataset

from cfg.datasets.BlenderGlass_dataset import safe_resize
from cfg.datasets.cleargrasp_dataset import mask_loader
from utils.exr_handler import png_depth_loader
from utils.rgbd2pcd import get_surface_normal_from_xyz, random_xyz_sampling, compute_xyz


class RealTimeCaptureDataset(Dataset):

    def __init__(self, root_dir, img_w=256, img_h=256,
                 mask_dir="mask", rgb_dir="rgb", depth_dir="depth_origin", depth_max=1.0,
                 depth_factor=4000, max_norm=True, sample=1, input_trajectory=None) -> None:
        super().__init__()
        self.base_dir = root_dir
        self.mask_dir = os.path.join(root_dir, mask_dir)
        self.rgb_dir = os.path.join(root_dir, rgb_dir)
        self.depth_dir = os.path.join(root_dir, depth_dir)
        self.input_trajectory = input_trajectory
        self.t_trajectory = []
        self.r_trajectory = []
        self.img_w = img_w
        self.img_h = img_h
        self.img_list = []
        self.mask_list = []
        self.depth_list = []
        self.img_name_list = []
        self.transform_seq = transforms.Compose(
            [
                transforms.Resize((self.img_h, self.img_w)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self._load_data()
        self.camera_origin = {
            "fx": 612.156,
            "fy": 610.956,
            "cx": 321.976,
            "cy": 246.287,
            "xres": 640,
            "yres": 480
        }
        self.camera = {}

        tmp = mask_loader(self.mask_list[0])
        h, w = tmp.shape
        scale = (self.img_h / h, self.img_w / w)
        self.camera["fy"] = self.camera_origin["fy"] * scale[0]
        self.camera["fx"] = self.camera_origin["fx"] * scale[1]
        self.camera["cy"] = self.camera_origin["cy"] * scale[0]
        self.camera["cx"] = self.camera_origin["cx"] * scale[1]
        self.camera["xres"] = self.img_w
        self.camera["yres"] = self.img_h
        self.depth_factor = depth_factor
        self.max_norm = max_norm
        self.sample = sample
        self.depth_max = depth_max
        self.sampled_points_num = (self.img_h // 4) * (self.img_w // 4)


    def _load_data(self):
        imgs = os.listdir(self.rgb_dir)
        imgs.sort(key=lambda x: x[:-4], reverse=False)

        for img in imgs:
            png_name = img[:-4]
            mask_png_name = png_name + ".pngmask" + ".png"
            img_path = os.path.join(self.rgb_dir, img)
            gt_png_path = os.path.join(self.depth_dir, img)
            mask_png_path = os.path.join(self.mask_dir, mask_png_name)
            if not os.path.exists(gt_png_path) or not os.path.exists(mask_png_path):
                raise ValueError("Error reading data:{}".format(img))
            self.img_list.append(img_path)
            self.depth_list.append(gt_png_path)
            self.mask_list.append(mask_png_path)
            self.img_name_list.append(png_name)

        if self.input_trajectory:
            with open(self.input_trajectory, "r") as fp:
                trajectory_list = fp.readlines()
                for tr in trajectory_list:
                    rq_vec = list(map(float, tr.strip().split()))
                    t_vec = np.expand_dims(np.array(rq_vec[1:4]), 1)
                    rq = Rotation.from_quat(rq_vec[4:]).as_matrix()
                    self.t_trajectory.append(t_vec)
                    self.r_trajectory.append(rq)
                fp.close()

    def __getitem__(self, idx):

        cur_rgb = Image.open(self.img_list[idx]).convert("RGB")
        cur_color = self.transform_seq(cur_rgb)

        # cur_rgb = safe_resize(np.array(cur_rgb), self.img_w, self.img_h).transpose(2, 0, 1) / 255

        original_mask = mask_loader(self.mask_list[idx])
        mask = safe_resize(original_mask, self.img_w, self.img_h)

        original_depth = png_depth_loader(self.depth_list[idx]) / self.depth_factor
        original_depth[original_depth > self.depth_max] = self.depth_max
        cur_depth = safe_resize(original_depth, self.img_w, self.img_h)
        xyz_img = compute_xyz(cur_depth, self.camera)
        cur_depth[np.isnan(cur_depth)] = 0
        cur_depth[np.isinf(cur_depth)] = 0
        # cur_depth[mask > 0] = 0

        zero_mask = (cur_depth < 0.1)
        zero_mask = np.logical_or(zero_mask, mask)

        # scale_max = np.amin(cur_depth[~(zero_mask > 0)]) * 10

        cur_xyz = compute_xyz(cur_depth, self.camera)  # before `norm`, used for icp
        if self.max_norm:
            depth_max = np.amax(cur_depth)
            # depth_max = 1.5
            cur_depth = cur_depth / depth_max
        else:
            depth_max = 1
        depth_min = np.amin(cur_depth[~(zero_mask > 0)])
        # tmp = depth_min * 10
        # depth_min = 0.15
        # cur_depth[cur_depth > tmp] = tmp

        # cur_depth[cur_depth > tmp] = 0

        depth_scale = torch.tensor(depth_max).repeat(self.img_h, self.img_w).float().unsqueeze(0)
        depth_min = torch.tensor(depth_min).repeat(self.img_h, self.img_w).float().unsqueeze(0)


        xyz_img = torch.from_numpy(xyz_img).permute(2, 0, 1).float()
        pt = random_xyz_sampling(xyz_img, n_points=self.sampled_points_num)

        if idx != len(self.img_list) - self.sample:
            forward_rgb = Image.open(self.img_list[idx+self.sample]).convert("RGB")
            forward_color = self.transform_seq(forward_rgb)
            # forward_rgb = safe_resize(np.array(forward_rgb), self.img_w, self.img_h).transpose(2, 0, 1) / 255
            forward_mask = mask_loader(self.mask_list[idx+self.sample])
            forward_mask = safe_resize(forward_mask, self.img_w, self.img_h)
            forward_depth = safe_resize(png_depth_loader(self.depth_list[idx+self.sample]) / self.depth_factor, self.img_w, self.img_h)
            forward_depth[np.isnan(forward_depth)] = 0
            forward_depth[np.isinf(forward_depth)] = 0
            forward_depth[forward_mask > 0] = 0
            forward_xyz = compute_xyz(forward_depth, self.camera)
        else:
            # forward_rgb = cur_rgb.copy()
            forward_color = cur_color.clone()
            forward_xyz = cur_xyz.copy()

        return {
            "cur_color": cur_color.float(),
            # "cur_rgb": torch.from_numpy(cur_rgb).float(),
            "raw_depth": torch.from_numpy(cur_depth).unsqueeze(0).float(),
            "depth_scale": depth_scale,
            "min_depth": depth_min,
            "mask": torch.from_numpy(mask).unsqueeze(0).float(),
            "zero_mask": torch.from_numpy(zero_mask).unsqueeze(0).float(),
            "pt": pt,
            # "forward_rgb": forward_rgb,
            "forward_color": forward_color.float(),
            "fx": torch.from_numpy(np.array(self.camera["fx"])).float(),
            "fy": torch.from_numpy(np.array(self.camera["fy"])).float(),
            "cx": torch.from_numpy(np.array(self.camera["cx"])).float(),
            "cy": torch.from_numpy(np.array(self.camera["cy"])).float(),

            "cur_xyz": cur_xyz,
            "forward_xyz": forward_xyz
        }

    def __len__(self):
        return len(self.img_list)
