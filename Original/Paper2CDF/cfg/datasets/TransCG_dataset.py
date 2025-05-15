import os
import json

import cv2
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset

from utils.data_augmentation import chromatic_transform, add_noise
from utils.rgbd2pcd import compute_xyz, get_surface_normal_from_xyz


class TransCG(Dataset):

    def __init__(self, data_dir, split='train', img_h=480, img_w=640):
        """
        Initialization.
        Parameters
        ----------
        data_dir: str, required, the data path;
        split: str in ['train', 'test'], optional, default: 'train', the dataset split option.
        """
        super(TransCG, self).__init__()
        if split not in ['train', 'test']:
            raise AttributeError('Invalid split option.')
        self.data_dir = data_dir
        self.split = split
        with open(os.path.join(self.data_dir, 'metadata.json'), 'r') as fp:
            self.dataset_metadata = json.load(fp)
        self.scene_num = self.dataset_metadata['total_scenes']
        self.perspective_num = self.dataset_metadata['perspective_num']
        self.scene_metadata = [None]
        for scene_id in range(1, self.scene_num + 1):
            with open(os.path.join(self.data_dir, 'scene{}'.format(scene_id), 'metadata.json'), 'r') as fp:
                self.scene_metadata.append(json.load(fp))
        self.total_samples = self.dataset_metadata['{}_samples'.format(split)]
        self.sample_info = []
        for scene_id in self.dataset_metadata[split]:
            scene_type = self.scene_metadata[scene_id]['type']
            scene_split = self.scene_metadata[scene_id]['split']
            assert scene_split == split, "Error in scene {}, expect split property: {}, found split property: {}.".format(
                scene_id, split, scene_split)
            for perspective_id in self.scene_metadata[scene_id]['D435_valid_perspective_list']:
                self.sample_info.append([
                    os.path.join(self.data_dir, 'scene{}'.format(scene_id), '{}'.format(perspective_id)),
                    1,  # (for D435)
                    scene_type
                ])
            for perspective_id in self.scene_metadata[scene_id]['L515_valid_perspective_list']:
                self.sample_info.append([
                    os.path.join(self.data_dir, 'scene{}'.format(scene_id), '{}'.format(perspective_id)),
                    2,  # (for L515)
                    scene_type
                ])
        # Integrity double-check
        assert len(
            self.sample_info) == self.total_samples, "Error in total samples, expect {} samples, found {} samples.".format(
            self.total_samples, len(self.sample_info))
        # Other parameters
        self.cam_intrinsics = [None,
                               np.load(os.path.join(self.data_dir, 'camera_intrinsics', '1-camIntrinsics-D435.npy')),
                               np.load(os.path.join(self.data_dir, 'camera_intrinsics', '2-camIntrinsics-L515.npy'))]
        self.DILATION_KERNEL = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).astype(np.uint8)
        self.img_w = img_w
        self.img_h = img_h
        self.depth_max = 1.5
        self.depth_min = 0.3
        self.depth_norm = 1.0
        self.use_aug = False
        self.rgb_aug_prob = 0.8

    def __getitem__(self, id):
        img_path, camera_type, scene_type = self.sample_info[id]
        rgb = np.array(Image.open(os.path.join(img_path, 'rgb{}.png'.format(camera_type))), dtype=np.float32)
        depth = np.array(Image.open(os.path.join(img_path, 'depth{}.png'.format(camera_type))), dtype=np.float32)
        depth_gt = np.array(Image.open(os.path.join(img_path, 'depth{}-gt.png'.format(camera_type))), dtype=np.float32)
        depth_gt_mask = np.array(Image.open(os.path.join(img_path, 'depth{}-gt-mask.png'.format(camera_type))),
                                 dtype=np.uint8)
        return self.process_data(rgb, depth, depth_gt, depth_gt_mask, self.cam_intrinsics[camera_type],
                                 scene_type=scene_type, camera_type=camera_type)

    def __len__(self):
        return self.total_samples

    def process_depth(self, depth, camera_type=0, depth_min=0.3, depth_max=1.5, depth_norm=1.0):
        """
        Process the depth information, including scaling, normalization and clear NaN values.
        Parameters
        ---------
        depth: array, required, the depth image;
        camera_type: int in [0, 1, 2], optional, default: 0, the camera type;
            - 0: no scale is applied;
            - 1: scale 1000 (RealSense D415, RealSense D435, etc.);
            - 2: scale 4000 (RealSense L515).
        depth_min, depth_max: int, optional, default: 0.3, 1.5, the min depth and the max depth;
        depth_norm: float, optional, default: 1.0, the depth normalization coefficient.
        Returns
        -------
        The depth image after scaling.
        """
        scale_coeff = 1
        if camera_type == 1:
            scale_coeff = 1000
        if camera_type == 2:
            scale_coeff = 4000
        depth = depth / scale_coeff
        depth[np.isnan(depth)] = 0.0
        depth = np.where(depth < depth_min, 0, depth)
        depth = np.where(depth > depth_max, 0, depth)
        depth = depth / depth_norm
        return depth

    def process_data(self, rgb, depth, depth_gt, depth_gt_mask, camera_intrinsics,
                     scene_type="cluttered", camera_type=0):

        rgb = cv2.resize(rgb, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
        depth_gt = cv2.resize(depth_gt, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
        depth_gt_mask = cv2.resize(depth_gt_mask, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
        depth_gt_mask = depth_gt_mask.astype(np.bool)

        # depth processing
        depth = self.process_depth(depth, camera_type=camera_type, depth_min=self.depth_min,
                                   depth_max=self.depth_max, depth_norm=self.depth_norm)
        depth_gt = self.process_depth(depth_gt, camera_type=camera_type, depth_min=self.depth_min,
                                      depth_max=self.depth_max, depth_norm=self.depth_norm)

        # RGB augmentation.
        if self.split == 'train' and self.use_aug and np.random.rand(1) > 1 - self.rgb_aug_prob:
            rgb = chromatic_transform(rgb)
            rgb = add_noise(rgb)

        # Geometric augmentation
        if self.split == 'train' and self.use_aug:
            has_aug = False
            if np.random.rand(1) > 0.5:
                has_aug = True
                rgb = np.flip(rgb, axis=0)
                depth = np.flip(depth, axis=0)
                depth_gt = np.flip(depth_gt, axis=0)
                depth_gt_mask = np.flip(depth_gt_mask, axis=0)
            if np.random.rand(1) > 0.5:
                has_aug = True
                rgb = np.flip(rgb, axis=1)
                depth = np.flip(depth, axis=1)
                depth_gt = np.flip(depth_gt, axis=1)
                depth_gt_mask = np.flip(depth_gt_mask, axis=1)
            if has_aug:
                rgb = rgb.copy()
                depth = depth.copy()
                depth_gt = depth_gt.copy()
                depth_gt_mask = depth_gt_mask.copy()

        # RGB normalization
        rgb = rgb / 255.0
        rgb = rgb.transpose(2, 0, 1)
        # process scene mask
        scene_mask = (scene_type == 'cluttered')

        # zero mask
        neg_zero_mask = np.where(depth_gt < 0.01, 255, 0).astype(np.uint8)
        neg_zero_mask_dilated = cv2.dilate(neg_zero_mask, kernel=self.DILATION_KERNEL)
        neg_zero_mask[neg_zero_mask != 0] = 1
        neg_zero_mask_dilated[neg_zero_mask_dilated != 0] = 1
        zero_mask = np.logical_not(neg_zero_mask)
        zero_mask_dilated = np.logical_not(neg_zero_mask_dilated)

        # loss mask
        initial_loss_mask = np.logical_and(depth_gt_mask, zero_mask)
        initial_loss_mask_dilated = np.logical_and(depth_gt_mask, zero_mask_dilated)
        if scene_mask:
            loss_mask = initial_loss_mask
            loss_mask_dilated = initial_loss_mask_dilated
        else:
            loss_mask = zero_mask
            loss_mask_dilated = zero_mask_dilated

        camera = {
            "fx": torch.tensor(camera_intrinsics[0, 0]),
            "fy": torch.tensor(camera_intrinsics[1, 1]),
            "cx": torch.tensor(camera_intrinsics[0, 2]),
            "cy": torch.tensor(camera_intrinsics[1, 2]),
            "xres": self.img_w,
            "yres": self.img_h
        }
        xyz_img = compute_xyz(depth, camera_intrinsics)
        xyz_gt = compute_xyz(depth_gt, camera_intrinsics)
        xyz_gt = torch.from_numpy(xyz_gt).permute(2, 0, 1).float()
        xyz_img = torch.from_numpy(xyz_img).permute(2, 0, 1).float()
        depth_gt_sn = get_surface_normal_from_xyz(xyz_gt.unsqueeze(0)).squeeze(0)

        data_dict = {
            'color': torch.FloatTensor(rgb),
            'raw_depth': torch.FloatTensor(depth),
            'gt_depth': torch.FloatTensor(depth_gt),
            'depth_gt_mask': torch.BoolTensor(depth_gt_mask),
            'mask': torch.tensor(depth_gt_mask),
            'loss_mask': torch.BoolTensor(loss_mask),
            'loss_mask_dilated': torch.BoolTensor(loss_mask_dilated),
            "depth_gt_sn": depth_gt_sn,
            "pt": xyz_img,
            "xyz_gt": xyz_gt
        }
        return data_dict, camera
