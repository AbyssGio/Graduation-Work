import json
import os
import cv2
import torch
import numpy as np
import glob

import torch.nn.functional as F
import yaml
from imgaug import augmenters as iaa
from skimage.transform import resize
from PIL import Image
from easydict import EasyDict as edict
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
from torchvision import transforms

from utils import exr_handler, rgbd2pcd
from utils.rgbd2pcd import get_surface_normal_from_xyz, random_xyz_sampling


def safe_resize(img, img_w, img_h):
    res = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    res[np.isnan(res)] = 0.0
    res[np.isinf(res)] = 0.0
    return res

def mask_loader(path_to_png):
    image = Image.open(path_to_png).convert("L")
    image = np.array(image) / 255.  # .transpose([2, 0, 1])
    mask = np.ones_like(image)
    # 0: background 1: object
    mask[np.where(image <= 0.01)] = 0
    return mask


def max_min(data):
    return (data - np.amin(data)) / (np.amax(data) - np.amin(data))


class ClearGrasp(Dataset):
    def __init__(
            self, root, split="train",
            img_h=480, img_w=640,
            specific_ds=None,
            max_norm=False,
            rgb_aug=False
    ):
        # Split can be `train` or `test`, `val`
        self.img_h = img_h
        self.img_w = img_w
        self.split = split
        assert split in ["val", "test", "train"]
        if split == "val" or split == 'test':
            self.data_root = os.path.join(root, 'cleargrasp-dataset-' + 'test-val')
        else:
            self.data_root = os.path.join(root, 'cleargrasp-dataset-' + split)
        self.color_name, self.depth_name, self.render_name, self.mask_name = [], [], [], []
        self.cam_parma_name = []
        self.specific_ds = specific_ds
        self._load_data()
        self.max_norm = max_norm
        self.rgb_aug = rgb_aug
        self.rgb_aug_prob = 0.5
        self.transform_seq = transforms.Compose(
            [
                transforms.Resize((self.img_h, self.img_w)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.dilation_kernel = np.ones((3, 3)).astype(np.uint8)

    def get_real_cam_params(self, yaml_file_path):
        with open(yaml_file_path, 'r') as f:
            intrinsics = edict(yaml.load(f, Loader=yaml.FullLoader))
        return intrinsics

    def get_syn_cam_params(self, json_file_path, img_size):
        # img_size: HxW
        meta_data = json.load(open(json_file_path, 'r'))
        # If the pixel is square, then fx=fy. Also note this is the cam params before scaling
        if 'camera' not in meta_data.keys() or 'field_of_view' not in meta_data['camera'].keys():
            fov_x = 1.2112585306167603
            fov_y = 0.7428327202796936
        else:
            fov_x = meta_data['camera']['field_of_view']['x_axis_rads']
            fov_y = meta_data['camera']['field_of_view']['y_axis_rads']
        if 'image' not in meta_data.keys():
            img_h = img_size[0]
            img_w = img_size[1]
        else:
            img_h = meta_data['image']['height_px']
            img_w = meta_data['image']['width_px']

        if 'camera' not in meta_data.keys() or 'world_pose' not in meta_data['camera'].keys() or \
                'rotation' not in meta_data['camera']['world_pose'].keys() or \
                'quaternion' not in meta_data['camera']['world_pose']['rotation'].keys():
            raise ValueError('No quaternion: {}'.format(json_file_path))
        else:
            q = meta_data['camera']['world_pose']['rotation']['quaternion']
            quaternion = np.array([q[1], q[2], q[3], q[0]])
            r = R.from_quat(quaternion)
            rot_from_q = r.as_matrix().astype(np.float32)
            world_pose = np.array(meta_data['camera']['world_pose']['matrix_4x4']).astype(np.float32)

        fx = img_w * 0.5 / np.tan(fov_x * 0.5)
        fy = img_h * 0.5 / np.tan(fov_y * 0.5)
        cx = img_w * 0.5
        cy = img_h * 0.5
        camera_params = {
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy,
            'yres': img_h,
            'xres': img_w,
            'world_pose': world_pose,
            'rot_mat': rot_from_q,
        }
        return camera_params

    def _load_data(self):

        # Check if split is test or validation
        split_type = self.split

        if split_type == 'train':
            models = os.listdir(self.data_root)
            for model in models:
                render_depth_f = sorted(glob.glob(os.path.join(self.data_root, model, 'depth-imgs-rectified', '*')))
                color_f = sorted(glob.glob(os.path.join(self.data_root, model, 'rgb-imgs', '*')))
                mask_f = sorted(glob.glob(os.path.join(self.data_root, model, 'segmentation-masks', '*')))
                json_f = [p.replace("rgb-imgs", "json-files").replace('-rgb.jpg', '-masks.json') for p in color_f]
                self.render_name += render_depth_f
                self.color_name += color_f
                self.depth_name += render_depth_f
                self.mask_name += mask_f
                self.cam_parma_name += json_f

        else:
            test_val_folders = os.listdir(self.data_root)
            if self.specific_ds:
                if "real" in self.specific_ds:
                    real_data = [self.specific_ds]
                    synthetic_data = []
                else:
                    real_data = []
                    synthetic_data = [self.specific_ds]
            elif split_type == 'val':
                real_data = ['real-val']
                synthetic_data = ['synthetic-val']
            else:
                real_data = ['real-test']
                synthetic_data = ['synthetic-test']
            # List of extensions
            EXT_COLOR_IMG = ['-transparent-rgb-img.jpg', '-rgb.jpg']  # '-rgb.jpg' - includes normals-rgb.jpg
            EXT_DEPTH_IMG = ['-depth-rectified.exr', '-transparent-depth-img.exr']
            EXT_DEPTH_GT = ['-depth-rectified.exr', '-opaque-depth-img.exr']
            EXT_MASK = ['-mask.png']

            for folder in test_val_folders:
                if folder in real_data:
                    for sub_folder in os.listdir(os.path.join(self.data_root, folder)):
                        json_file_path = os.path.join(self.data_root, folder, sub_folder, 'camera_intrinsics.yaml')
                        for ext in EXT_COLOR_IMG:
                            color_f = sorted(glob.glob(os.path.join(self.data_root, folder, sub_folder, '*' + ext)))
                            self.color_name += color_f
                            self.cam_parma_name += ([json_file_path] * len(color_f))
                        for ext in EXT_DEPTH_IMG:
                            depth_f = sorted(glob.glob(os.path.join(self.data_root, folder, sub_folder, '*' + ext)))
                            self.depth_name += depth_f
                        for ext in EXT_DEPTH_GT:
                            render_depth_f = sorted(
                                glob.glob(os.path.join(self.data_root, folder, sub_folder, '*' + ext)))
                            self.render_name += render_depth_f
                        for ext in EXT_MASK:
                            mask_f = sorted(glob.glob(os.path.join(self.data_root, folder, sub_folder, '*' + ext)))
                            self.mask_name += mask_f
                elif folder in synthetic_data:
                    models = os.listdir(os.path.join(self.data_root, folder))
                    for model in models:
                        render_depth_f = sorted(
                            glob.glob(os.path.join(self.data_root, folder, model, 'depth-imgs-rectified', '*')))
                        color_f = sorted(glob.glob(os.path.join(self.data_root, folder, model, 'rgb-imgs', '*')))
                        mask_f = sorted(
                            glob.glob(os.path.join(self.data_root, folder, model, 'segmentation-masks', '*')))
                        json_f = [p.replace("rgb-imgs", "json-files").replace('-rgb.jpg', '-masks.json') for p in color_f]
                        self.render_name += render_depth_f
                        self.color_name += color_f
                        self.depth_name += render_depth_f
                        self.mask_name += mask_f
                        self.cam_parma_name += json_f

    def __len__(self):
        return len(self.depth_name)

    def __getitem__(self, index):
        # read data from files
        color = Image.open(self.color_name[index]).convert("RGB")
        render_depth = exr_handler.exr_loader(self.render_name[index], ndim=1)
        assert len(render_depth.shape) == 2, 'There is channel dimension'
        raw_depth = exr_handler.exr_loader(self.depth_name[index], ndim=1)
        # clean NaN and INF value
        render_depth[np.isnan(render_depth)] = 0.0
        render_depth[np.isinf(render_depth)] = 0.0
        # Load the mask
        mask = mask_loader(self.mask_name[index])
        if self.depth_name[index].endswith('depth-rectified.exr'):
            # Remove the portion of the depth image with transparent object
            # If image is synthetic
            raw_depth[np.where(mask > 0)] = 0
        # restrict depth region to [0.3, 1.5]
        # render_depth = np.where(render_depth < 0.3, 0, render_depth)
        # render_depth = np.where(render_depth > 1.5, 0, render_depth)
        if self.max_norm:
            depth_ma = np.amax(raw_depth)
            raw_depth = raw_depth / depth_ma
            # raw_depth = np.where(raw_depth < 0.2, 0, raw_depth)
        else:
            depth_ma = 1.0
            # raw_depth = np.where(raw_depth < 0.3, 0, raw_depth)
            # raw_depth = np.where(raw_depth > 1.5, 0, raw_depth)
        # Load camera intrinsics
        cam_param_file = self.cam_parma_name[index]
        if cam_param_file.endswith(".json"):
            camera_param = self.get_syn_cam_params(cam_param_file, (color.size[1], color.size[0]))
        elif cam_param_file.endswith(".yaml"):
            camera_param = self.get_real_cam_params(cam_param_file)
        else:
            raise Exception("No camera params found in file: %s" % cam_param_file)
        camera_intrinsics = {
            "fx": camera_param["fx"],
            "fy": camera_param["fy"],
            "cx": camera_param["cx"],
            "cy": camera_param["cy"],
            "xres": camera_param["xres"],
            "yres": camera_param["yres"]
        }
        # compute pt
        xyz_img = rgbd2pcd.compute_xyz(raw_depth, camera_intrinsics)
        xyz_gt = rgbd2pcd.compute_xyz(render_depth, camera_intrinsics)
        # scale affect fx, fy, cx, cy
        img_size = color.size  # wxh
        scale = (img_size[0] / self.img_w, img_size[1] / self.img_h)
        camera_intrinsics["fx"] *= scale[0]
        camera_intrinsics["fy"] *= scale[1]
        camera_intrinsics["cx"] *= scale[0]
        camera_intrinsics["cy"] *= scale[1]
        camera_intrinsics["xres"] = self.img_w
        camera_intrinsics["yres"] = self.img_h

        cur_rgb = color.copy()
        cur_rgb = safe_resize(np.array(cur_rgb), self.img_w, self.img_h).transpose(2, 0, 1) / 255
        cur_rgb = torch.from_numpy(cur_rgb).float()
        # data augmentation and resize
        color = self.transform_seq(color)
        # depth gt resize
        render_depth = cv2.resize(render_depth, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
        render_depth[np.isnan(render_depth)] = 0.0
        render_depth[np.isinf(render_depth)] = 0.0

        # depth input resize
        raw_depth = cv2.resize(raw_depth, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
        raw_depth[np.isnan(raw_depth)] = 0.0
        raw_depth[np.isinf(raw_depth)] = 0.0

        # resize mask
        mask = safe_resize(mask, self.img_w, self.img_h)

        depth_mi = np.amin(raw_depth[~(mask > 0)])
        # numpy data to torch data
        raw_depth = torch.from_numpy(raw_depth).unsqueeze(0).float()
        render_depth = torch.from_numpy(render_depth).unsqueeze(0).float()
        # resize point cloud
        xyz_img = cv2.resize(xyz_img, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
        xyz_img = torch.from_numpy(xyz_img).permute(2, 0, 1).float()
        xyz_gt = cv2.resize(xyz_gt, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
        xyz_gt = torch.from_numpy(xyz_gt).permute(2, 0, 1).float()  # 3xHxW

        # sn mask
        sn_mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        sn_mask = cv2.erode(sn_mask, kernel=self.dilation_kernel)
        sn_mask[sn_mask != 0] = 1
        # sn_mask = np.logical_not(sn_mask)
        sn_mask = np.logical_and(sn_mask, mask)

        # composed mask: object mask + gt mask(zero mask)
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        sn_mask = torch.from_numpy(sn_mask).unsqueeze(0).float()
        # surface normal gt
        depth_gt_sn = get_surface_normal_from_xyz(xyz_gt.unsqueeze(0)).squeeze(0)
        # ignore zero point and sampling
        pt = random_xyz_sampling(xyz_img, 4800)
        depth_scale = torch.tensor(depth_ma).repeat(self.img_h, self.img_w).float().unsqueeze(0)
        min_depth = torch.tensor(depth_mi).repeat(self.img_h, self.img_w).float().unsqueeze(0)
        return {
            "cur_color": color,
            "cur_rgb": cur_rgb,
            "raw_depth": raw_depth,
            "depth_scale": depth_scale,
            "min_depth": min_depth,
            "mask": mask,
            "sn_mask": sn_mask,
            "pt": pt,
            "cur_gt_depth": render_depth,
            "depth_gt_sn": depth_gt_sn,
            "R_mat": torch.eye(3).float(),
            "t_vec": torch.zeros(3).float(),
            "forward_rgb": cur_rgb.clone(),
            "forward_color": color.clone(),
            "forward_depth": render_depth.clone(),
            "fx": torch.tensor(np.array(camera_intrinsics["fx"])).float(),
            "fy": torch.tensor(np.array(camera_intrinsics["fy"])).float(),
            "cx": torch.tensor(np.array(camera_intrinsics["cx"])).float(),
            "cy": torch.tensor(np.array(camera_intrinsics["cy"])).float(),
        }
