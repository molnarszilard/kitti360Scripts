import numpy as np
import os
import kapture
import kitti
import kapture
import numpy as np
import quaternion as quat
from kitti360scripts.helpers.project import CameraPerspective as Camera


def world2cam(poses: dict):
    keys = list(poses.keys())
    cam = {}
    for key in keys:
        P = poses[key]
        R = P[:3, :3]
        t = P[:3, -1]

        c = np.linalg.inv(P)

        cam[key] = c
    return cam

# root to images
kitti_root = "/mnt/cuda_external_5TB/datasets/kitti/kitti360/KITTI-360/"
sequence = "2013_05_28_drive_0010_sync"
cameraID = "image_00"
kitti_images_root = os.path.join(kitti_root,"data_2d_raw",sequence,cameraID,"data_rect")
poses_path = os.path.join(kitti_root,"data_poses",sequence,"cam0_to_world.txt")

result_folder="/mnt/ssd2/datasets/kitti360_temp/"



# intrinsic calibration file
yaml_path = "../../datasets/kitti/calibration/image_02.yaml"

# target folder with all images
target_root = os.path.join(result_folder,sequence,"kapture/kitti/gt/full")

# target folder with a subset from the images
target_filtered_root = os.path.join(result_folder,sequence,"kapture/kitti/gt/filtered")

###
pose_dict = kitti.parse_poses(poses_path)
keys = list(pose_dict.keys())

camera = Camera(root_dir=kitti_root, seq=sequence)
world_pos = camera.cam2world
cam_pose  = world2cam(world_pos)
world_quat = kitti.pose_to_quaternion(cam_pose)


kapture_sensors_root = os.path.abspath(os.path.join(target_root, "sensors"))
os.makedirs(kapture_sensors_root, exist_ok=True)
kapture.write_trajectories(world_quat, "0", kapture_sensors_root)             # type: ignore

all_images = os.listdir(kitti_images_root)
print(kitti_images_root)
os.makedirs(os.path.join(kapture_sensors_root, "records_data"), exist_ok=True)

for img in all_images:
    target_img = os.path.abspath(os.path.join(kapture_sensors_root, "records_data", img))
    if not os.path.exists(target_img):
        os.symlink(os.path.abspath(os.path.join(kitti_images_root, img)), target_img)

# kapture.kitti_to_kapture_sensors(yaml_path, kapture_sensors_root)

# kapture.create_records_camera(kapture_sensors_root, kapture_sensors_root)

# os.makedirs(target_filtered_root, exist_ok=True)

# kapture.filter_dataset_by_keeping(kapture_sensors_root, target_filtered_root, keep=4)

kapture.make_images_list_filtered(
    os.path.abspath(os.path.join(target_root, "sensors/records_data")),
    os.path.join(target_filtered_root, "images_list.txt"),
    keep=4
)
