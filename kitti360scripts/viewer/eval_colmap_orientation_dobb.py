
#Python code to create teh DOBB dataset from Kitti360 
# importing the required modules
import os
from kitti360scripts.helpers.annotation  import Annotation3D
import cv2
import numpy as np  
import math
from kitti360scripts.helpers.project import CameraPerspective as Camera
import argparse
from scipy.spatial.transform import Rotation
import geometry_utils
import matplotlib.pyplot as plt
import pandas as pd
import csv

np.set_printoptions(suppress=True, precision=6)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--kitti_root',default='/mnt/cuda_external_5TB/datasets/kitti/kitti360/KITTI-360/',
                    help='the root of the kitti folder of original data')
parser.add_argument('--sequence', default='2013_05_28_drive_0010_sync',
                    help='the sequence')
parser.add_argument('--cameraID', default='image_00',
                    help='default camera ID')
parser.add_argument('--result_folder',default='/mnt/ssd2/datasets/kitti360_pose3_veh_build/',
                    help='the root folder of the results')
parser.add_argument('--list', default='/mnt/ssd2/datasets/kitti360_pose3_veh_build/2013_05_28_drive_0010_sync/frames_class_both.txt',  #'/mnt/ssd2/datasets/kitti360_pose2_veh_build/2013_05_28_drive_0010_sync/frames_class_both.txt'
                    help='path to a txt file, which contains the frames that should be considered in the comparison')
args = parser.parse_args()

camera = Camera(root_dir=args.kitti_root, seq=args.sequence)
csv_labels_folder=os.path.join(args.cameraID,'ellipse_dir_data_pred/')

### Creating Output structure
base=os.path.join(args.result_folder,args.sequence)
colmap_results_path=os.path.join(args.kitti_root,'colmap_results',args.sequence+'_kapture.txt')
colmap_loclist_path=os.path.join(args.kitti_root,'colmap_results',args.sequence+'_kapture_localization.txt')
new_csv_path='cm_rmatrices'

def plot_rotated_axes(ax, r, name=None, offset=(0, 0, 0), scale=1):
    colors = ("#FF6666", "#005533", "#1199EE")  # Colorblind-safe RGB
    loc = np.array([offset, offset])
    for i, (axis, c) in enumerate(zip((ax.xaxis, ax.yaxis, ax.zaxis),colors)):
        axlabel = axis.axis_name
        axis.set_label_text(axlabel)
        axis.label.set_color(c)
        axis.line.set_color(c)
        axis.set_tick_params(colors=c)
        line = np.zeros((2, 3))
        line[1, i] = scale
        line_rot = r.apply(line)
        line_plot = line_rot + loc
        ax.plot(line_plot[:, 0], line_plot[:, 1], line_plot[:, 2], c)
        text_loc = line[1]*1.2
        text_loc_rot = r.apply(text_loc)
        text_plot = text_loc_rot + loc[0]
        ax.text(*text_plot, axlabel.upper(), color=c,
                va="center", ha="center")
    ax.text(*offset, name, color="k", va="center", ha="center",
            bbox={"fc": "w", "alpha": 0.8, "boxstyle": "circle"})

### Comparing two 3 dimensional vectors
### From https://stackoverflow.com/a/13849249
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def read_csv(frame,camera_R):
    min_R = np.zeros_like(camera_R)
    min_angle = math.pi*2
    with open(os.path.join(base,csv_labels_folder, f'{frame:010d}.csv'), newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=';', quotechar='|')
        counter = -1
        for row in csvreader:
            counter+=1
            if counter==0:
                continue
            # Rc2w = np.array([[row[15],row[16],row[17]],[row[18],row[19],row[20]],[row[21],row[22],row[23]]]).astype(np.float64)
            dirX=float(row[13])
            dirY=float(row[14])
            theta=float(row[15])
            rotY=float(row[16])
            rotX=float(row[17])
            dpt_angle = math.atan2(dirY,dirX)
            phiZ = theta-dpt_angle
            rotZwimu = np.asarray(Rotation.from_euler('Z', phiZ, degrees=False).as_matrix())
            rotYwimu = np.asarray(Rotation.from_euler('Y', rotY, degrees=False).as_matrix())
            rotXwimu = np.asarray(Rotation.from_euler('X', rotX, degrees=False).as_matrix())
            Rc2w = rotZwimu@rotYwimu@rotXwimu
            ref=np.array([1.0,0.0,0.0])
            dobbX = np.matmul(Rc2w,ref)
            cameraX = np.matmul(camera_R,ref)
            ang_diff = angle_between(dobbX,cameraX)
            if ang_diff<min_angle:
                min_angle=ang_diff
                min_R=Rc2w
    return min_R

def qvec2rotmat(qvec):
    ### https://github.com/colmap/colmap/blob/main/scripts/python/read_write_model.py#L524 visited on 26.02.2025.
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )    

def process():
    frames2check = []
    if args.list is not None:
        f_frames = open(args.list, "r")
        Lines_frames = f_frames.readlines()
        f_frames.close()
        for line in Lines_frames:
            current_line = line[:-1]
            elements=current_line.split(" ")
            frames2check.append(int(elements[0]))
        frames2check=np.asarray(frames2check)
    else:
        exit()
    f_label = open(colmap_loclist_path, "r")
    Lines_colmap_loclist = f_label.readlines()
    f_label.close()
    locframes_colmap = []
    for line in Lines_colmap_loclist:
        locframes_colmap.append(int(os.path.splitext(line[:-1])[0]))
    f_label = open(colmap_results_path, "r")
    Lines_colmap = f_label.readlines()
    f_label.close()
    line_number=-1
    number_of_frames = 0
    for line in Lines_colmap:
        if line.startswith('#'):
            continue
        current_line = line[:-1]
        elements=current_line.split(" ")
        frame = int(os.path.splitext(elements[9])[0])
        if frame>number_of_frames:
            number_of_frames=frame
    colmap_results=np.zeros((number_of_frames+1,8))
    for line in Lines_colmap:
        if line.startswith('#'):
                continue
        line_number+=1
        if line_number%2:
            continue
        current_line = line[:-1]
        elements=current_line.split(" ")
        image_ID = int(elements[0]) # I am not sure what it is, or why it does not match with the image name
        qw = float(elements[1])
        qx = float(elements[2])
        qy = float(elements[3])
        qz = float(elements[4])
        tx = float(elements[5])
        ty = float(elements[6])
        tz = float(elements[7])
        camera_ID = int(elements[8]) # using only 1 camera, so should be 1 all the time
        # if camera_ID!=1:
        #     print("Camera is not 1, but %d"%(camera_ID))
        frame = int(os.path.splitext(elements[9])[0])
        # colmap_results.append([frame,qw,qx,qy,qz,tx,ty,tz])
        if not frame in locframes_colmap:
            colmap_results[frame]=frame,qw,qx,qy,qz,tx,ty,tz

    # colmap_results=sorted(colmap_results, key=lambda x: x[0])
    colmap_results=np.asarray(colmap_results)
    new_csv = []
    row=[]
    row.append('ImageName')
    row.append('angle_diff_cm')
    row.append('angle_diff_GT')
    row.append('angle_diff_dobb')
    row.append('cm_error_rel')
    row.append('dobb_error_rel')
    row.append('cm_error_abs')
    row.append('dobb_error_abs')
    row.append('Rc2w-11')
    row.append('Rc2w-12')
    row.append('Rc2w-13')
    row.append('Rc2w-21')
    row.append('Rc2w-22')
    row.append('Rc2w-23')
    row.append('Rc2w-31')
    row.append('Rc2w-32')
    row.append('Rc2w-33')
    new_csv.append(row)
    frame=-1
    first_frame=True
    for frame in frames2check:
        if len(colmap_results)>frame:
            colmap_estimate = colmap_results[frame]
        else:
            continue
        if colmap_estimate[0]==0:
            continue

        row = []
        row.append(f'{frame:010d}.png')
        _frame_colmap,qw,qx,qy,qz,tx,ty,tz=colmap_estimate
        if frame!=_frame_colmap:
            print("something wrong")
        qvec = [qw,qx,qy,qz]
        Rcolmap=qvec2rotmat(qvec).T
        Tcolmap=np.array([tx,ty,tz])
     

        valid_key_found = False
        valid_key = frame
        while not valid_key_found:
            try:
                camera_tr = camera.cam2world[valid_key]
                valid_key_found=True
            except:
                valid_key-=1

        ### camera_tr --> Tr(cam_0 -> world)
        # camera_tr = camera.cam2world[frame]
        camera_R = camera_tr[:3, :3]
        camera_T = camera_tr[:3, 3]
        dobb_R = read_csv(frame,camera_R)
        if first_frame:
            first_frame=False
            Rcolmap_prev=Rcolmap
            camera_R_prev=camera_R
            dobb_R_prev= dobb_R

        ref=np.array([1.0,0.0,0.0])
        colmap_prev = np.matmul(Rcolmap_prev,ref)
        colmap_current = np.matmul(Rcolmap,ref)
        camera_prev = np.matmul(camera_R_prev,ref)
        camera_current = np.matmul(camera_R,ref)
        dobb_prev = np.matmul(dobb_R_prev,ref)
        dobb_current = np.matmul(dobb_R,ref)
        ang_diff_colmap = angle_between(colmap_prev,colmap_current)
        ang_diff_camera = angle_between(camera_prev,camera_current)
        ang_diff_dobb = angle_between(dobb_prev,dobb_current)

        ang_diff_current_camera = angle_between(camera_current,colmap_current)
        ang_diff_current_dobb = angle_between(camera_current,dobb_current)

        error_cm = abs(math.atan2(math.sin(ang_diff_colmap - ang_diff_camera),math.cos(ang_diff_colmap - ang_diff_camera)))
        error_dobb = abs(math.atan2(math.sin(ang_diff_dobb - ang_diff_camera),math.cos(ang_diff_dobb - ang_diff_camera)))

        print('Frame: %d, -> cm: %f, cam: %f, dobb: %f'%(frame,math.degrees(ang_diff_colmap),math.degrees(ang_diff_camera),math.degrees(ang_diff_dobb)))
        row.append(math.degrees(ang_diff_colmap))
        row.append(math.degrees(ang_diff_camera))
        row.append(math.degrees(ang_diff_dobb))
        row.append(math.degrees(error_cm))
        row.append(math.degrees(error_dobb))
        row.append(math.degrees(ang_diff_current_camera))
        row.append(math.degrees(ang_diff_current_dobb))
        Rcolmap_1D = np.reshape(Rcolmap,9)
        for elem in Rcolmap_1D:
            row.append(elem)
        new_csv.append(row)
        Rcolmap_prev=Rcolmap
        camera_R_prev=camera_R
        dobb_R_prev= dobb_R
    df = pd.DataFrame(np.asarray(new_csv))
    df.to_csv(os.path.join(base,'colmap_rmatrices.csv',),header=False, index=False, sep=';', quotechar='|')
              

def main():     
        process()
      
if __name__ == "__main__": 
    main() 
