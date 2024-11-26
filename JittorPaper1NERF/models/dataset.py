# from numpy.core.fromnumeric import squeeze
import jittor
from jittor import nn as jnn
import cv2 as cv
import numpy as np
import os
from glob import glob
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import json

np.random.seed(0)
jittor.flags.use_cuda = 1


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    # 这里可能是想从文件加载P，但这里已经有P了
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    # 解码透视投影矩阵，这里P应该是个摄影机传输进来的图像生成的矩阵。接受4x4的三维投影矩阵
    out = cv.decomposeProjectionMatrix(P)
    # K=旋转矩阵（相机坐标系的旋转），R=平移向量（相机到原点的平移），t=内部参数矩阵（包含了相机透镜等参数）
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    # 生成4x4的单精度浮点数单位矩阵，然后做一些看不懂的变换（？？？）
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')

        jittor.flags.use_cuda = 1
        # self.device = torch.device('cuda')

        self.conf = conf

        # 从设置文件里调取各种参数
        self.data_dir = conf.get_string('data_dir')
        self.render_cameras_name = conf.get_string('render_cameras_name')
        self.object_cameras_name = conf.get_string('object_cameras_name')
        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

        # 拼接路径，适配操作系统
        camera_dict = os.path.join(self.data_dir, self.render_cameras_name)
        self.camera_dict = camera_dict
        # 加载图片列表，这里的image_list应该是一个包括所有数据图片的文件名列表，而后记录数据个数
        self.images_lis = sorted(glob(os.path.join(self.data_dir, 'images/*.jpg')))
        self.n_images = len(self.images_lis)
        # 这里利用np.stack函数将图片都堆叠起来，成为一个np数组，每个元素都是一个由cv2生成的图片数据矩阵，同时生成遮罩矩阵
        self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 256.0
        self.masks_np = np.stack([np.ones_like(im[:, :, 0]) for im in self.images_np])

        self.intrinsics_all = []
        self.pose_all = []
        dict_all = {}
        # 读取java script格式的文件，在womask.conf文件里能找到路径（可是没有数据集我怎么测试呢
        with open(self.camera_dict, 'r') as f:
            dict_all = json.loads(f.read())

        for i in range(self.n_images):
            # 记录图片的名字（编号
            img_name = self.images_lis[i].split('/')[-1]
            # K中是一个4x4的单精度浮点数矩阵
            K = np.array(dict_all[img_name]['K']).reshape(4, 4).astype(np.float32)
            W2C = np.array(dict_all[img_name]['W2C']).reshape(4, 4).astype(np.float32)
            P = K @ W2C
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            # 这里应该是逐张照片进行处理，把每个照片所包含的摄像信息用4x4矩阵的方式存储在_all里面，这里用Var存储float32
            self.intrinsics_all.append(jittor.array(intrinsics).float())
            self.pose_all.append(jittor.array(pose).float())

        # 已全部转换为jittor的var类型变量
        self.images = jittor.array(self.images_np.astype(np.float32))  # .cpu()  # [n_images, H, W, 3]
        self.masks = jittor.array(self.masks_np.astype(np.float32))  # .cpu()  # [n_images, H, W, 3]
        self.intrinsics_all = jittor.stack(self.intrinsics_all)  # .to(self.device)  # [n_images, 4, 4]
        self.intrinsics_all_inv = jittor.linalg.inv(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = jittor.stack(self.pose_all)  # .to(self.device)  # [n_images, 4, 4]
        self.H, self.W = self.images.shape()[1], self.images.shape()[2]
        self.image_pixels = self.H * self.W

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([1.01, 1.01, 1.01, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        self.object_bbox_min = object_bbox_min[:3]
        self.object_bbox_max = object_bbox_max[:3]
        print('Load data: End')
        # 数据处理完毕

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = jittor.linspace(0, self.W - 1, self.W // l)
        ty = jittor.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = jittor.meshgrid(tx, ty)
        p = jittor.stack([pixels_x, pixels_y, jittor.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = jittor.nn.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / jittor.norm(p, p=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = jittor.nn.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = jittor.randint(low=0, high=self.W, shape=[batch_size])
        pixels_y = jittor.randint(low=0, high=self.H, shape=[batch_size])
        color = self.images[img_idx][(pixels_y, pixels_x)]  # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]  # batch_size, 3
        p = jittor.stack([pixels_x, pixels_y, jittor.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = jnn.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze()  # batch_size, 3
        rays_v = p / jittor.norm(p, p=2, dim=-1, keepdim=True)  # batch_size, 3
        rays_v = jnn.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape)  # batch_size, 3
        return jittor.concat([rays_o.cpu(), rays_v.cpu(), color, mask.unsqueeze(-1)], dim=-1)  # batch_size, 10

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = jittor.linspace(0, self.W - 1, self.W // l)
        ty = jittor.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = jittor.meshgrid(tx, ty)
        p = jittor.stack([pixels_x, pixels_y, jittor.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = jnn.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / jittor.norm(p, p=2, dim=-1, keepdim=True)  # W, H, 3
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
        rot = jittor.array(pose[:3, :3]).cuda()
        trans = jittor.array(pose[:3, 3]).cuda()
        rays_v = jnn.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d):
        a = jittor.Var.sum(rays_d ** 2, dim=-1, keepdims=True)
        b = jittor.array(2.0) * jittor.Var.sum(rays_o * rays_d, dim=-1, keepdims=True)
        mid = jittor.array(0.5 * (-b)) / a
        near = mid - jittor.array(1.0)
        far = mid + jittor.array(1.0)
        return near, far

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)