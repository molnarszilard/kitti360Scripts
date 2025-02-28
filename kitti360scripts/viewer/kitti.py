import os
import numpy as np
import quaternion as quat


def pose_to_quaternion(pose_dict: dict):
    keys = list(pose_dict.keys())
    pose_quat = {}
    for k in keys:
        pose_quat[k] = {
            "q": quat.from_rotation_matrix(pose_dict[k][:3,:3]),
            "t": pose_dict[k][:3, -1]
        }

    return pose_quat


def parse_calibration(path, img_id=0):
    with open(path, "r") as f:
        data = f.readlines()

    for line in data:
        img = line.split(":")[0]
        curr_id = int(img.split("_")[-1])
        if curr_id == img_id:
            line_split = line.strip().split(": ")[-1].split(" ")
            data = map(float, line_split)
            return np.concatenate((np.array(list(data)).reshape((3, 4)), np.array([0., 0., 0., 1.]).reshape(1, 4)))


def parse_poses(path):

    pose_dict = {}

    poses = np.loadtxt(path)
    
    idxs = poses[:, 0]

    transforms = poses[:, 1:].reshape((-1, 3, 4))

    for i in range(len(idxs)):
        pose_dict[idxs[i]] = np.concatenate((transforms[i], np.array([0.,0.,0.,1.]).reshape(1,4)))

    return pose_dict


def cam2world(poses: dict, calibration):
    keys = list(poses.keys())
    world = {}
    for key in keys:
        world[key] = np.matmul(poses[key], calibration)
    return world

# def world2cam(poses: dict):
#     keys = list(poses.keys())
#     cam = {}
#     for key in keys:
#         P = poses[key]
        
#         cam[key] = 
