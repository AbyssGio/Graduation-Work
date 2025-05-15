from functools import total_ordering
import numpy as np
import torch
import torch.nn.functional as F
import einops
import open3d as o3d
import copy


def rgbd2pcd(rgb_image_path, depth_image_path):
    color_raw = o3d.io.read_image(rgb_image_path)
    depth_raw = o3d.io.read_image(depth_image_path)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
        )
    )
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return pcd


def compute_xyz(depth_img, camera_params):
    """ Compute ordered point cloud from depth image and camera parameters.

        If focal lengths fx,fy are stored in the camera_params dictionary, use that.
        Else, assume camera_params contains parameters used to generate synthetic data (e.g. fov, near, far, etc)

        @param depth_img: a [H x W] numpy array of depth values in meters
        @param camera_params: a dictionary with parameters of the camera used
    """

    # Compute focal length from camera parameters
    fx = camera_params['fx']
    fy = camera_params['fy']
    x_offset = camera_params['cx']
    y_offset = camera_params['cy']
    # indices = np.indices((camera_params['yres'], camera_params['xres']), dtype=np.float32).transpose(1, 2, 0)
    indices = np.indices((depth_img.shape[0], depth_img.shape[1]), dtype=np.float32).transpose(1, 2, 0)
    z_e = depth_img
    x_e = (indices[..., 1] - x_offset) * z_e / fx
    y_e = (indices[..., 0] - y_offset) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1)  # Shape: [H x W x 3]
    return xyz_img


def gradient(x):
    """
    Get gradient of pt image.
    This is adapted from implicit-depth repository, ref: https://github.com/NVlabs/implicit_depth/blob/main/src/utils/point_utils.py.
    Parameters
    ----------
    x: the pt map to get gradient.
    Returns
    -------
    the x-axis-in-image gradient and y-axis-in-image gradient of the pt map.
    """
    left = x
    right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
    top = x
    bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]
    dx, dy = right - left, bottom - top
    dx[:, :, :, -1] = 0
    dy[:, :, -1, :] = 0
    return dx, dy


def get_surface_normal_from_xyz(x, epsilon=1e-8, method="1-point", **kwargs):
    """
    Get the surface normal of pt image in Pytorch 
    ----------
    Parameters
    x: the pt map to get surface normal;
    epsilon: float, optional, default: 1e-8, the epsilon to avoid nan.
    Returns
    method: choose algorithm in [`1-points`, `lsq`, `kdtree`]
    -------
    Return
    The surface normals.
    """
    if method == "1-point":
        """
        This is adapted from implicit-depth repository, 
        ref: https://github.com/NVlabs/implicit_depth/blob/main/src/utils/point_utils.py
        """
        dx, dy = gradient(x)
        surface_normal = torch.cross(dx, dy, dim=1)
        surface_normal = surface_normal / (torch.norm(surface_normal, dim=1, keepdim=True) + epsilon)

    elif method == "kdtree":
        # radius = kwargs.get("radius", 0.1)
        max_nn = kwargs.get("max_nn", 30)
        b = x.size()[0]
        pcd = o3d.geometry.PointCloud()
        res = []
        for batch_idx in range(b):
            pt = x[batch_idx, :, :, :].view(3, -1).transpose(1, 0)
            pcd.points = o3d.utility.Vector3dVector(pt.detach().cpu().numpy())
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamKNN(knn=max_nn)
            )
            res.append(np.asarray(pcd.normals))
        res = np.array(res)
        surface_normal = torch.from_numpy(res).to(x.device)
        surface_normal = surface_normal.reshape(x.size())

    else:
        raise NotImplementedError("Unsupport method: {}".format(method))

    return surface_normal


def get_xyz(depth, fx, fy, cx, cy):
    """
    Get XYZ from depth image and camera intrinsics.
    Parameters
    ----------
    depth: tensor, required, the depth image;
    fx, fy, cx, cy: tensor, required, the camera intrinsics;
    Returns
    -------
    The XYZ value of each pixel.
    """
    bs, _, h, w = depth.shape

    # h_idx = torch.arange(h).unsqueeze(1).repeat(1, w)
    # w_idx = torch.arange(w).unsqueeze(0).repeat(h, 1)
    # indices = torch.stack([h_idx, w_idx], dim=0).repeat(bs, 1, 1, 1)
    # indices = indices.float().to(depth.device)
    indices = np.indices((h, w), dtype=np.float32)
    indices = torch.FloatTensor(np.array([indices] * bs)).to(depth.device)
    z = depth.squeeze(1)
    x = (indices[:, 1, :, :] - einops.repeat(cx, 'bs -> bs h w', h=h, w=w)) * z / einops.repeat(fx, 'bs -> bs h w', h=h,
                                                                                                w=w)
    y = (indices[:, 0, :, :] - einops.repeat(cy, 'bs -> bs h w', h=h, w=w)) * z / einops.repeat(fy, 'bs -> bs h w', h=h,
                                                                                                w=w)
    return torch.stack([x, y, z], dim=1)


def get_xyz_cpp(depth, fx: float, fy: float, cx: float, cy: float):
    bs, _, h, w = depth.shape
    h_idx = torch.arange(h).unsqueeze(1).repeat(1, w)
    w_idx = torch.arange(w).unsqueeze(0).repeat(h, 1)
    indices = torch.stack([h_idx, w_idx], dim=0).repeat(bs, 1, 1, 1)
    indices = indices.float().to(depth.device)
    z = depth.squeeze(1)
    x = (indices[:, 1, :, :] - cx) * z / fx
    y = (indices[:, 0, :, :] - cy) * z / fy
    return torch.stack([x, y, z], dim=1)


def get_surface_normal_from_depth(depth, fx, fy, cx, cy, epsilon=1e-8):
    """
    Get surface normal from depth and camera intrinsics.
    Parameters
    ----------
    depth: tensor, required, the depth image;
    fx, fy, cx, cy: tensor, required, the camera intrinsics;
    epsilon: float, optional, default: 1e-8, the epsilon to avoid nan.
    Returns
    -------
    The surface normals.
    """
    xyz = get_xyz(depth, fx, fy, cx, cy)
    return get_surface_normal_from_xyz(xyz, epsilon=epsilon)


def uniform_xyz_sampling(xyz, every_p=10):
    """
    :param xyz:
    :param every_p:
    :return:
    """
    pcd = o3d.geometry.PointCloud()
    # if torch.is_tensor(xyz):
    #     pcd.points = o3d.utility.Vector3dVector(xyz.numpy())
    # else:
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.uniform_down_sample(every_p)
    return np.asarray(pcd.points)


def random_xyz_sampling(xyz, n_points=4096):
    """
    :param xyz: (3xN or 3xHxW) original point cloud
    :param n_points: (S) sampled point number
    :param mask: sampled firstly in mask region
    :return: (3xS) sampled point cloud
    """
    if len(xyz.size()) == 3:
        pt = xyz.view(3, -1).transpose(0, 1)  # -> (Nx3)
    else:
        pt = xyz.transpose(0, 1)
    non_zeros = torch.abs(torch.sum(pt, dim=1)).gt(1e-6)
    pt_non_zeros = pt[non_zeros]
    n_pts = pt_non_zeros.size()[0]
    if n_pts < n_points:
        rnd_idx = torch.cat([torch.randint(0, n_pts, (n_points,))])
    else:
        rnd_idx = torch.randperm(n_pts)[:n_points]
    res = pt_non_zeros[rnd_idx, :]
    return res.transpose(0, 1)


def fixedNumDownSample(vertices, desiredNumOfPoint, leftVoxelSize, rightVoxelSize):
    """ Use the method voxel_down_sample defined in open3d and do bisection iteratively
        to get the appropriate voxel_size which yields the points with the desired number.
        INPUT:
            vertices: numpy array shape (n,3)
            desiredNumOfPoint: int, the desired number of points after down sampling
            leftVoxelSize: float, the initial bigger voxel size to do bisection
            rightVoxelSize: float, the initial smaller voxel size to do bisection
        OUTPUT:
            downSampledVertices: down sampled points with the original data type

    """
    assert leftVoxelSize > rightVoxelSize, "leftVoxelSize should be larger than rightVoxelSize"
    assert vertices.shape[
               0] > desiredNumOfPoint, "desiredNumOfPoint should be less than or equal to the num of points in the given array."
    if vertices.shape[0] == desiredNumOfPoint:
        return vertices

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd = pcd.voxel_down_sample(leftVoxelSize)
    assert len(pcd.points) <= desiredNumOfPoint, "Please specify a larger leftVoxelSize."
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd = pcd.voxel_down_sample(rightVoxelSize)
    assert len(pcd.points) >= desiredNumOfPoint, "Please specify a smaller rightVoxelSize."

    pcd.points = o3d.utility.Vector3dVector(vertices)
    midVoxelSize = (leftVoxelSize + rightVoxelSize) / 2.
    pcd = pcd.voxel_down_sample(midVoxelSize)
    while len(pcd.points) != desiredNumOfPoint:
        if len(pcd.points) < desiredNumOfPoint:
            leftVoxelSize = copy.copy(midVoxelSize)
        else:
            rightVoxelSize = copy.copy(midVoxelSize)
        midVoxelSize = (leftVoxelSize + rightVoxelSize) / 2.
        pcd.points = o3d.utility.Vector3dVector(vertices)
        pcd = pcd.voxel_down_sample(midVoxelSize)

    # print("final voxel size: ", midVoxelSize)
    downSampledVertices = np.asarray(pcd.points, dtype=vertices.dtype)
    return downSampledVertices, leftVoxelSize, rightVoxelSize
