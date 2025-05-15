import os
from os import listdir
from os.path import isfile, join
import sys
import re
import argparse
import subprocess

from utils.png2klg import write_klg


def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def timeFile(fileList, subfolder, drop_num=0):
    count = 0.033333
    data_dict = []
    for idx, fileName in enumerate(fileList):
        if idx >= (len(fileList) - drop_num):
            continue
        data_dict.append((count, "%s/%s" % (subfolder, fileName)))
        count = count + 0.033333
    return dict(data_dict)



def read_file_list(filename):
    """
    Reads a trajectory from a text file.

    File format:
    The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp (to be matched)
    and "d1 d2 d3.." is arbitary data (e.g., a 3D position and 3D orientation) associated to this timestamp.

    Input:
    filename -- File name

    Output:
    dict -- dictionary of (stamp,data) tuples

    """
    file = open(filename)
    data = file.read()
    lines = data.replace(",", " ").replace("\t", " ").split("\n")
    list = [[v.strip() for v in line.split(" ") if v.strip() != ""] for line in lines if
            len(line) > 0 and line[0] != "#"]
    list = [(float(l[0]), l[1:]) for l in list if len(l) > 1]
    return dict(list)


def associate(first_list, second_list, offset, max_difference):
    """
    Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim
    to find the closest match for every input tuple.

    Input:
    first_list -- first dictionary of (stamp,data) tuples
    second_list -- second dictionary of (stamp,data) tuples
    offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
    max_difference -- search radius for candidate generation

    Output:
    matches -- list of matched tuples ((stamp1,data1),(stamp2,data2))

    """
    first_keys = list(first_list.keys())
    second_keys = list(second_list.keys())
    potential_matches = [(abs(a - (b + offset)), a, b)
                         for a in first_keys
                         for b in second_keys
                         if abs(a - (b + offset)) < max_difference]
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))

    matches.sort()
    return matches


# python run_elasticfusion.py --base_dir=/home/ctwo/blender_glass_data/val_seq4
# --depth_file=depth --rgb_file=rgb --mask_file=mask
# --drop_num=1 --depth_scale=4000

if __name__ == '__main__':

    # parse command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", help="file that contain 'depth', 'rgb', 'mask")
    parser.add_argument("--depth_file", help='first text file (format: timestamp data)', default="depth")
    parser.add_argument("--rgb_file", help='second text file (format: timestamp data)', default="rgb")
    parser.add_argument("--mask_file", help='second text file (format: timestamp data)', default="mask")
    parser.add_argument("--drop_num", default=0, type=int, help="drop last n file ordered by timestamp")
    parser.add_argument('--offset', help='time offset added to the timestamps of the second file (default: 0.0)',
                        default=0.0)
    parser.add_argument('--max_difference',
                        help='maximally allowed time difference for matching entries (default: 0.02)', default=0.02)
    # parser.add_argument("--png2klg", help="path of executable file 'png2klg'")
    parser.add_argument("--depth_scale", help="scale factor of depth", type=int, default=4000)
    # parser.add_argument("--e_fusion", help="path of executable file of elasticfusion")
    args = parser.parse_args()

    base_dir = args.base_dir
    depth_path = os.path.join(base_dir, args.depth_file)
    rgb_path = os.path.join(base_dir, args.rgb_file)
    mask_path = os.path.join(base_dir, args.mask_file)
    drop_num = args.drop_num
    depth_scale = args.depth_scale
    offset = args.offset
    max_difference = args.max_difference

    rgb_files = [f for f in listdir(rgb_path) if isfile(join(rgb_path, f))]
    depth_files = [f for f in listdir(depth_path) if isfile(join(depth_path, f))]
    mask_files = [os.path.join(args.mask_file, f) for f in listdir(mask_path) if isfile(join(mask_path, f))]
    mask_files.sort()

    intersection = list(set(rgb_files) & set(depth_files))
    intersection.sort(key=natural_key)

    rgb_data = timeFile(intersection, args.rgb_file, 0)
    depth_data = timeFile(intersection, args.depth_file, 0)

    matches = associate(depth_data, rgb_data, offset, max_difference)

    with open(os.path.join(base_dir, "associations.txt"), "w+") as fp:
        for a, b in matches:
            fp.write("%f %s %f %s\n" % (a, depth_data[a], b - float(args.offset), rgb_data[b]))
            print("%f %s %f %s" % (a, depth_data[a], b - float(args.offset), rgb_data[b]))
    fp.close()
    #
    # rgb_imgs = []
    # depth_imgs = []
    # mask_imgs = []
    # timestamps = []
    # for idx, (a, b) in enumerate(matches):
    #     depth_imgs.append(depth_data[a])
    #     rgb_imgs.append(rgb_data[b])
    #     mask_imgs.append(mask_files[idx])
    #     timestamps.append(a)
    # klg_path = os.path.join(base_dir, "scene_v2.klg")
    # camera_path = os.path.join(base_dir, "camera")
    # with open(camera_path, "r") as fp:
    #     fx, fy, cx, cy = fp.readline().split()
    # camera = {
    #     "fx": float(fx),
    #     "fy": float(fy),
    #     "cx": float(cx),
    #     "cy": float(cy)
    # }
    # write_klg(base_dir, rgb_imgs, depth_imgs, mask_imgs, timestamps, camera, klg_path, drop_num, depth_scale)

