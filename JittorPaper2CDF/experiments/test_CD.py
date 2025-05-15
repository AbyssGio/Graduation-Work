import os

import numpy as np
import open3d as o3d


def CD_distance(pred_ply, gt_ply):
    # pred_mesh = o3d.io.read_triangle_mesh(pred_ply)
    # gt_mesh = o3d.io.read_triangle_mesh(gt_ply)
    # pred_pcd = o3d.geometry.PointCloud()
    # pred_pcd.points = o3d.utility.Vector3dVector(pred_mesh.vertices)
    # gt_pcd = o3d.geometry.PointCloud()
    # gt_pcd.points = o3d.utility.Vector3dVector(gt_mesh.vertices)
    pred_pcd = o3d.io.read_point_cloud(pred_ply)
    gt_pcd = o3d.io.read_point_cloud(gt_ply)

    dists = gt_pcd.compute_point_cloud_distance(pred_pcd)
    dists = np.asarray(dists)

    return dists.mean()


base_dir = "/home/ctwo/blender_glass_data"
name_prefix = ".klg.ply"
gt_ply_name = "scene_origin" + name_prefix
compared_method = ["sn", "cg", "trans"]
scene_list = ["seq9"]
for scene in scene_list:
    scene_path = os.path.join(base_dir, scene)
    gt_ply_path = os.path.join(scene_path, gt_ply_name)
    for m in compared_method:
        predict_ply_name = "scene_" + m + name_prefix
        predict_ply_path = os.path.join(scene_path, predict_ply_name)
        print("============= {} ==============".format(m))
        print(CD_distance(predict_ply_path, gt_ply_path))
        print("===========================")

