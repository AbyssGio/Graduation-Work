import cv2
import matplotlib.pyplot as plt
import numpy as np
import jittor as jt
from matplotlib import animation
# import pangolin
# import OpenGL.GL as gl
import open3d as o3d
import os
from utils.rgbd2pcd import get_surface_normal_from_depth, get_surface_normal_from_xyz, compute_xyz


def vis_depth(depth_array, mi=None, ma=None, color_map='RdBu_r', visualize=True, save_path=None):
    if not mi:
        vmin = np.amin(depth_array)
    else:
        vmin = mi
    if ma:
        vmax = np.amax(depth_array)
    else:
        vmax = ma
    plt.imshow(depth_array, cmap=color_map, vmin=vmin, vmax=vmax)
    plt.axis("off")
    if visualize:
        plt.show()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.clf()


def animated_frames(frames, v_min, v_max, figsize=(10, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    im = ax.imshow(frames[0], cmap='RdBu_r', vmin=v_min, vmax=v_max)

    def animate(i):
        im.set_array(frames[i])
        return [im, ]

    ani = animation.FuncAnimation(fig, animate, frames=len(frames),
                                  interval=60, blit=True)

    return ani


def animated_scatter(frames, trajs, figsize=(10, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    im = ax.imshow(frames[0])
    scat = ax.scatter(trajs[0][:, 1], trajs[0][:, 0],
                      facecolors='none', edgecolors='r')

    def animate(i):
        im.set_array(frames[i])
        if len(trajs[i]) > 0:
            scat.set_offsets(trajs[i][:, [1, 0]])
        else:  # If no trajs to draw
            scat.set_offsets([])  # clear the scatter plot

        return [im, scat, ]

    ani = animation.FuncAnimation(fig, animate, frames=len(frames),
                                  interval=60, blit=True)

    return ani


def vis_mask(mask, img=None, visualize=True, save_path=None):
    if img:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img = cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
        # img = img[:, :, ::-1]
        # img[..., 2] = np.where(mask == 1, 255, img[..., 2])
        plt.imshow(img)

    else:
        plt.imshow(mask, "gray", vmin=0, vmax=1)

    plt.axis("off")
    if visualize:
        plt.show()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.clf()


def vis_normal(nmap):
    surface_normal = nmap.float().squeeze().cpu().numpy()
    # print(nmap.shape)
    surface_normal = surface_normal.transpose(1, 2, 0)  # (-1, 1)

    # print(np.amax(surface_normal))
    surface_normal[surface_normal > 1] = 1.0
    surface_normal[surface_normal < -1] = -1.0
    plt.axis("off")
    plt.imshow((surface_normal + 1) / 2, cmap="RdBu_r", vmin=0, vmax=1)
    plt.show()
    plt.clf()


def vis_surface_normal(depth, camera, color_map='Greys', visualize=True, save_path=None):
    xyz_img = compute_xyz(depth, camera)
    xyz_img = jt.array(xyz_img).float().unsqueeze(0).permute(0, 3, 1, 2)
    surface_normal = get_surface_normal_from_xyz(xyz_img)
    surface_normal = surface_normal.squeeze().cpu().numpy()
    surface_normal = surface_normal.transpose(1, 2, 0)  # (-1, 1)
    plt.axis("off")
    plt.imshow((surface_normal + 1) / 2, cmap=color_map, vmin=0, vmax=1)
    if visualize:
        plt.show()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.clf()


def vis_pt_3d(pt, visualize=True, save_path=None, v_size=None):
    if pt.shape[0] == 3:
        pt = pt.swapaxes(0, 1)
    if len(pt.shape) == 2 and pt.shape[1] == 3:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pt)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        if v_size:
            pcd = pcd.voxel_down_sample(voxel_size=v_size)
        if save_path:
            o3d.io.write_point_cloud("result.pcd", pcd)
        if visualize:
            o3d.visualization.draw_geometries([pcd])
    else:
        raise ValueError("`pt` must have 2-dimension: (n, 3)")


def vis_surface_normal_3d(pt, v_size=0.05, radius=0.5):
    if pt.shape[0] == 3:
        pt = pt.swapaxes(0, 1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pt)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # down sample and visualize normal
    if v_size:
        pcd = pcd.voxel_down_sample(voxel_size=v_size)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    o3d.visualization.draw_geometries([pcd], point_show_normal=True)


def make_dir(path, dir_name):
    dir_path = os.path.join(path, dir_name)
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        pass
    else:
        os.mkdir(dir_path)
        print("Successfully make dir {}".format(dir_path))
    return dir_path
