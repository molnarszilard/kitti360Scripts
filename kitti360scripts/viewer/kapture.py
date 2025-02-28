import os
import yaml


def write_trajectories(poses: dict, device_id: str, output_path: str):
    '''
    Write a trajectories.txt file from a poses dict. The dict should have another dictionary 
    with the 'q' and 't' keys. The keys of the initial dictionary will be used as timestamps
    '''
    
    keys = list(poses.keys())

    with open(os.path.join(output_path, "trajectories.txt"), "w") as f:
        f.write("# kapture format: 1.1\n")
        f.write("# timestamp, device_id, qw, qx, qy, qz, tx, ty, tz\n")
        for k in keys:
            q = poses[k]['q']
            tx, ty, tz = poses[k]['t']
            timestamp = "%010d" % k
            f.write(
                f"{timestamp}, {device_id}, {q.w}, {q.x}, {q.y}, {q.z}, {tx}, {ty}, {tz}\n"
            )
    print("Done!")



def kitti_to_kapture_sensors(yaml_path, output_path, cam_type="MEI"):
    if cam_type == "MEI":
        kitti_to_kapture_sensors_mei(yaml_path, output_path)
    elif cam_type == "CYLINDRICAL":
        kitti_to_kapture_sensors_cyl(yaml_path, output_path)
    

def kitti_to_kapture_sensors_mei(yaml_path, output_path):
    '''
    The MEI camera model is not implemented by default in either COLMAP or Kapture. The custom 
    implementation needs the params in the following order:
        
        w, h, gamma1, gamma2, cx, cy, xi, k1, k2

    Note that gamma1 and gamma2 can be thought of as fx and fy as they contain them, along other info.  
    '''
    
    with open(yaml_path, 'r') as f:
        f.readline()
        intrinsics = yaml.safe_load(f)
    
    w = intrinsics["image_width"]
    h = intrinsics["image_height"]
    xi = intrinsics["mirror_parameters"]["xi"]
    k1 = intrinsics["distortion_parameters"]["k1"]
    k2 = intrinsics["distortion_parameters"]["k2"]
    gamma1 = intrinsics["projection_parameters"]["gamma1"]
    gamma2 = intrinsics["projection_parameters"]["gamma2"]
    cx = intrinsics["projection_parameters"]["u0"]
    cy = intrinsics["projection_parameters"]["v0"]

    with open(os.path.join(output_path, "sensors.txt"), 'w') as f:
        f.write("# kapture format: 1.1\n")
        f.write("# sensor_device_id, name, sensor_type, [sensor_params]+\n")
        f.write(f"0, , camera, MEI_FISHEYE, {w}, {h}, {gamma1}, {gamma2}, {cx}, {cy}, {xi}, {k1}, {k2}")


def kitti_to_kapture_sensors_cyl(yaml_path, output_path):
    
    with open (os.path.join(output_path, "sensors.txt"), 'w') as f:
        f.write("# kapture format: 1.1\n")
        f.write("# sensor_device_id, name, sensor_type, [sensor_params]+\n")
        f.write("0, , camera, CYLINDRICAL, 1400, 2000, 401.1258395409511, 221.7076074690947, 700.0, 1000.0583903374148")
    


def create_records_camera(kapture_sensors_path, output_path):
    '''
    The input path should already contain the sensors/records_data folder with images.
    The file 'sensors.txt' must also be present in the sensors folder.

        timestamp, device_id, image_path
    '''

    all_images = os.listdir(os.path.join(kapture_sensors_path, "records_data"))
    all_images = sorted([img for img in all_images if img.endswith((".png", ".jpg"))]) # type: ignore
    
    with open(os.path.join(kapture_sensors_path, "trajectories.txt")) as f:
        lines = f.readlines()

    data_dict = {}
    for line in lines:
        if line.startswith("#"):
            continue
        data = line.split(", ")
        data_dict[int(data[0])] = data[1]

    # not all images have a timestamp in trajectories.txt
    # the images are named with the timestamps in the %010d format

    keys = list(map(int, data_dict.keys()))
    
    with open(os.path.join(kapture_sensors_path, "records_camera.txt"), "w") as f:
        f.write("# kapture format: 1.1\n")
        f.write("# timestamp, device_id, image_path\n")
        for img in all_images:
            img_id = int(img.split(".")[0])
            if img_id in keys:
                timestamp = "%010d" % img_id
                f.write(f"{timestamp}, {data_dict[img_id]}, {img}\n")


def filter_dataset_by_keeping(kapture_root: str, output: str, keep=2):
    '''
    Reduce the number of considered images in a kapture dataset by keeping every x entry. 
    The images are symlinked, the trajectories.txt and records_camera.txt files are recreated
    and the sensors.txt file is copied as is.
    
    Args:
        kapture_root: path to the sensors folder (with records_data/, sensors.txt etc.)
        output: where to put the new files. Images are Symlinked.
        percent: (=50.0) the percent of the images to retain
    '''
    import shutil
    
    if not output.strip().endswith("sensors") or not output.strip().endswith("sensors/"):
        output = os.path.join(output, "sensors")
        os.makedirs(output, exist_ok=True)

    shutil.copy(os.path.join(kapture_root, "sensors.txt"), output)

    os.makedirs(os.path.join(output, "records_data"), exist_ok=True)


    with open(os.path.join(kapture_root, "records_camera.txt"), 'r') as f:
        records_camera_data = f.readlines()
        
    with open(os.path.join(kapture_root, "trajectories.txt"), 'r') as f:
        trajectories_data = f.readlines()
        
    rc = open(os.path.join(output, "records_camera.txt"), 'w')
    tr = open(os.path.join(output, "trajectories.txt"), 'w')

    rc.write("# kapture format: 1.1\n")
    rc.write("# timestamp, device_id, image_path\n")

    tr.write("# kapture format: 1.1\n")
    tr.write("# timestamp, device_id, qw, qx, qy, qz, tx, ty, tz\n")

    cnt = 0
    for i in range(len(records_camera_data)):
        if records_camera_data[i].startswith("#"):
            continue

        if cnt % keep == 0:
            rc.write(records_camera_data[i])
            tr.write(trajectories_data[i])
            try:
                img_name = records_camera_data[i].strip().split(", ")[-1]
                os.symlink(os.path.realpath(os.path.join(kapture_root, 'records_data', img_name)), os.path.join(output, "records_data", img_name))
            except FileExistsError:
                continue

        cnt += 1

    rc.close()
    tr.close()


def make_images_list_filtered(images_root, out_file, keep=2):
    all_images = sorted(os.listdir(images_root))
    with open(out_file, "w") as f:
        for i in range(len(all_images)):
            if i % keep == 0:
                # f.write(os.path.join(images_root, all_images[i]) + "\n")
                f.write(all_images[i] + "\n")
