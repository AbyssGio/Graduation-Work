
import cv2
import numpy as np
import jittor as jt


from cfg.datasets.BlenderGlass_dataset import BlenderGlass # Y
from cfg.datasets.cleargrasp_dataset import safe_resize  # Y
from utils import rgbd2pcd # D Y
from utils.icp import icp_o3d # D Y
from utils.rgbd2pcd import get_surface_normal_from_xyz, random_xyz_sampling # D Y


class BlenderGlassV2(BlenderGlass):

    def __init__(self, root, scene="scene1", img_h=480, img_w=640, rgb_aug=False, max_norm=True, depth_factor=1000,
                 forward_windows_size=1, depth_max=10):
        self.forward_windows_size = forward_windows_size
        super().__init__(root, scene, img_h, img_w, rgb_aug, max_norm, depth_factor, depth_max)

    def __len__(self):
        return len(self.img_list)  # ignore last data

    def __getitem__(self, index):
        # assert index < (len(self.img_list) - self.forward_windows_size)
        cur_rgb, cur_mask, cur_render_depth, cur_raw_depth = self._get_item(index)

        # depth norm
        if self.max_norm:
            depth_ma = np.amax(cur_raw_depth)
            cur_raw_depth = cur_raw_depth / depth_ma
        else:
            depth_ma = 1.0
        depth_mi = np.amin(cur_raw_depth[~(cur_mask > 0)])

        cur_pose = self.pose_list[index]
        color = self.transform_seq(cur_rgb)
        xyz_img = rgbd2pcd.compute_xyz(cur_raw_depth, self.camera)
        xyz_gt = rgbd2pcd.compute_xyz(cur_render_depth, self.camera)
        xyz_img = jt.array(xyz_img).permute(2, 0, 1)
        xyz_img.float32()
        xyz_gt = jt.array(xyz_gt).permute(2, 0, 1).float32()  # 3xHxW

        sn_mask = np.where(cur_mask > 0, 255, 0).astype(np.uint8)
        sn_mask = cv2.erode(sn_mask, kernel=self.dilation_kernel)
        sn_mask[sn_mask != 0] = 1
        # sn_mask = np.logical_not(sn_mask)
        sn_mask = np.logical_and(sn_mask, cur_mask)

        sn_mask = jt.array(sn_mask).unsqueeze(0).float()
        cur_mask = jt.array(cur_mask).unsqueeze(0).float()
        val_mask = (cur_render_depth < self.depth_max)
        val_mask = np.logical_and(val_mask, cur_mask)

        cur_raw_depth = jt.array(cur_raw_depth).unsqueeze(0).float()
        cur_render_depth = jt.array(cur_render_depth).unsqueeze(0).float()
        cur_rgb = safe_resize(np.array(cur_rgb), self.img_w, self.img_h).transpose(2, 0, 1) / 255
        cur_rgb = jt.array(cur_rgb).float()
        depth_gt_sn = get_surface_normal_from_xyz(xyz_gt.unsqueeze(0)).squeeze(0)
        pt = random_xyz_sampling(xyz_img, self.sampled_points_num)
        training_mask = jt.ones_like(cur_mask)

        data_dict = {
            "cur_color": color,
            # "cur_rgb": cur_rgb,
            "raw_depth": cur_raw_depth,
            "depth_scale": jt.array(depth_ma).repeat(self.img_h, self.img_w).float().unsqueeze(0),
            "min_depth": jt.array(0.01).repeat(self.img_h, self.img_w).float().unsqueeze(0),
            "mask": cur_mask,
            "training_mask": training_mask,
            "val_mask": val_mask,
            "pt": pt,
            # "xyz_gt": xyz_gt,
            "cur_gt_depth": cur_render_depth,
            "depth_gt_sn": depth_gt_sn,
            "fx": jt.array(np.array(self.camera["fx"])).float(),
            "fy": jt.array(np.array(self.camera["fy"])).float(),
            "cx": jt.array(np.array(self.camera["cx"])).float(),
            "cy": jt.array(np.array(self.camera["cy"])).float(),
        }
        if self.forward_windows_size == 0:
            forward_rgb, forward_mask, forward_depth, _ = self._get_item(index)
            trans_mat = np.diag([1, 1, 1, 1])
            forward_color = self.transform_seq(forward_rgb)
            data_dict["forward_{}_color".format(1)] = forward_color
            data_dict["forward_{}_R_mat".format(1)] = jt.array(trans_mat[:3, :3]).float()
            data_dict["forward_{}_t_vec".format(1)] = jt.array(trans_mat[:3, 3]).float()
        else:
            for i in range(1, self.forward_windows_size+1):
                if (index + i) < len(self.img_list):
                    forward_rgb, forward_mask, forward_depth, _ = self._get_item(index + i)
                    forward_pose = self.pose_list[index + i]
                    trans_mat = np.matmul(np.linalg.inv(forward_pose), cur_pose)
                else:
                    forward_rgb, forward_mask, forward_depth, _ = self._get_item(index)
                    trans_mat = np.diag([1, 1, 1, 1])
                forward_color = self.transform_seq(forward_rgb)

                if not self.with_trajectory:
                    icp_depth = cur_render_depth.copy()
                    icp_depth[cur_mask > 0] = 0
                    icp_cur_xyz = rgbd2pcd.compute_xyz(icp_depth, self.camera)
                    icp_cur_xyz = icp_cur_xyz.reshape(-1, 3)
                    icp_depth = forward_depth.copy()
                    icp_depth[forward_mask > 0] = 0
                    icp_forward_xyz = rgbd2pcd.compute_xyz(icp_depth, self.camera)
                    icp_forward_xyz = icp_forward_xyz.reshape(-1, 3)
                    trans_mat = icp_o3d(icp_cur_xyz, icp_forward_xyz, tolerance=0.1)
                data_dict["forward_{}_color".format(i)] = forward_color.clone()
                data_dict["forward_{}_R_mat".format(i)] = jt.array(trans_mat[:3, :3]).float()
                data_dict["forward_{}_t_vec".format(i)] = jt.array(trans_mat[:3, 3]).float()
            # data_dict["forward_{}_rgb".format(i)] = forward_rgb
            # data_dict["forward_{}_depth".format(i)] = forward_depth

        return data_dict
