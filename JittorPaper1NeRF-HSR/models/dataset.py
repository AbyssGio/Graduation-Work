from numpy.core.fromnumeric import squeeze
import jittor as jt
import cv2 as cv
import numpy as np
import os
from glob import glob
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import json
from outerjittor.sqz import Osqueeze

np.random.seed(0)


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


class Dataset(jt.dataset.Dataset):
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        #self.device = jt.device('cuda')
        self.conf = conf

        self.data_dir = conf.get_string('data_dir')
        self.render_cameras_name = conf.get_string('render_cameras_name')
        self.object_cameras_name = conf.get_string('object_cameras_name')

        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

        camera_dict = os.path.join(self.data_dir, self.render_cameras_name)
        self.camera_dict = camera_dict
        self.images_lis = sorted(glob(os.path.join(self.data_dir, 'images/*.jpg')))
        self.n_images = len(self.images_lis)
        self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 256.0
        self.masks_np = np.stack([np.ones_like(im[:, :, 0]) for im in self.images_np])

        self.intrinsics_all = []
        self.pose_all = []
        dict_all = {}
        with open(self.camera_dict, 'r') as f:
            dict_all = json.loads(f.read())

        for i in range(self.n_images):
            img_name = self.images_lis[i].split('/')[-1]
            img_name = img_name.split('\\')[-1]
            K = np.array(dict_all[img_name]['K']).reshape(4, 4).astype(np.float32)
            W2C = np.array(dict_all[img_name]['W2C']).reshape(4, 4).astype(np.float32)

            P = K @ W2C
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(jt.array(intrinsics).float())
            self.pose_all.append(jt.array(pose).float())

        self.images = jt.array(self.images_np.astype(np.float32)) # [n_images, H, W, 3]
        self.masks = jt.array(self.masks_np.astype(np.float32)) # [n_images, H, W, 3]
        self.intrinsics_all = jt.stack(self.intrinsics_all)  # [n_images, 4, 4]
        self.intrinsics_all_inv = jt.linalg.inv(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = jt.stack(self.pose_all)  # [n_images, 4, 4]
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([1.01, 1.01, 1.01, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        self.object_bbox_min = object_bbox_min[:3]
        self.object_bbox_max = object_bbox_max[:3]

        print('Load data: End')

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = jt.linspace(0, self.W - 1, self.W // l)
        ty = jt.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = jt.meshgrid(tx, ty)
        p = jt.stack([pixels_x, pixels_y, jt.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = jt.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None])  # W, H, 3
        p = Osqueeze(p)
        rays_v = p / jt.norm(p, p=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = jt.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None])  # W, H, 3
        rays_v = Osqueeze(rays_v)
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = jt.randint(low=0, high=self.W, shape=[batch_size])
        pixels_y = jt.randint(low=0, high=self.H, shape=[batch_size])
        color = Osqueeze(self.images[img_idx])[(pixels_y, pixels_x)]   # batch_size, 3
        mask = Osqueeze(self.masks[img_idx])[(pixels_y, pixels_x)]   # batch_size, 3
        p = jt.stack([pixels_x, pixels_y, jt.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = jt.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None])# batch_size, 3
        p = Osqueeze(p)
        rays_v = p / jt.norm(p, p=2, dim=-1, keepdim=True)  # batch_size, 3
        rays_v = jt.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]) # batch_size, 3
        rays_v = Osqueeze(rays_v)
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape)  # batch_size, 3
        rays_o = Osqueeze(rays_o)
        return jt.concat([rays_o, rays_v, color, mask.unsqueeze(-1)], dim=-1)  # batch_size, 10

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = jt.linspace(0, self.W - 1, self.W // l)
        ty = jt.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = jt.meshgrid(tx, ty)
        p = jt.stack([pixels_x, pixels_y, jt.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = jt.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None])  # W, H, 3
        p = Osqueeze(p)
        rays_v = p / jt.norm(p, p=2, dim=-1, keepdim=True)  # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = jt.array(pose[:3, :3]).cuda()
        trans = jt.array(pose[:3, 3]).cuda()
        rays_v = jt.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None])  # W, H, 3
        rays_v = Osqueeze(rays_v)
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d):
        a = jt.sum(rays_d ** 2, dim=-1, keepdims=True)
        b = 2.0 * jt.sum(rays_o * rays_d, dim=-1, keepdims=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)
