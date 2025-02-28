
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
parser.add_argument('--result_folder',default='/mnt/ssd2/datasets/kitti360_temp/',
                    help='the root folder of the results')
parser.add_argument('--list', default='/mnt/ssd2/datasets/kitti360_pose2_veh_build/2013_05_28_drive_0010_sync/frames_class_both.txt',  #'/mnt/ssd2/datasets/kitti360_pose2_veh_build/2013_05_28_drive_0010_sync/frames_class_both.txt'
                    help='path to a txt file, which contains the frames that should be considered in the comparison')
args = parser.parse_args()

camera = Camera(root_dir=args.kitti_root, seq=args.sequence)
csv_labels_folder='ellipse_dir_data_pred/'

### Creating Output structure
base=os.path.join(args.result_folder,args.sequence)
orbslam_results_path=os.path.join(args.kitti_root,'orbslam_results',args.sequence+'204.txt')
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

def process():
    f_label = open(orbslam_results_path, "r")
    Lines_orbslam = f_label.readlines()
    f_label.close()
    orbslam_results=[]
    line_number=-1
    for line in Lines_orbslam:
        if line.startswith('#'):
                continue
        line_number+=1
        if line_number%2:
            continue
        current_line = line[:-1]
        elements=current_line.split(" ")
        R00 = float(elements[0])
        R01 = float(elements[1])
        R02 = float(elements[2])
        R03 = float(elements[3])
        R10 = float(elements[4])
        R11 = float(elements[5])
        R12 = float(elements[6])
        R13 = float(elements[7])
        R20 = float(elements[8])
        R21 = float(elements[9])
        R22 = float(elements[10])
        R23 = float(elements[11])
        orbslam_results.append([R00,R01,R02,R03,R10,R11,R12,R13,R20,R21,R22,R23])

    # orbslam_results=sorted(orbslam_results, key=lambda x: x[0])
    orbslam_results=np.asarray(orbslam_results)
    new_csv = []
    row=[]
    row.append('ImageName')
    row.append('angle_diff_orb')
    row.append('angle_diff_camera')
    row.append('angle_diff')
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
    for orbslam_estimate in orbslam_results:
        frame+=1
        if frame==0:# I do not know what to do with image 0000000000.png as it is not annotated
            continue
        # if frame%10:
        #     continue
        row = []
        row.append(f'{frame:010d}.png')
        R00,R01,R02,R03,R10,R11,R12,R13,R20,R21,R22,R23=orbslam_estimate
        Rorbslam=np.array([[R00,R01,R02],[R10,R11,R12],[R20,R21,R22]])        

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
        if first_frame:
            first_frame=False
            Rorbslam_prev=Rorbslam
            camera_R_prev=camera_R

        ref=np.array([1.0,0.0,0.0])
        orbslam_prev = np.matmul(Rorbslam_prev,ref)
        orbslam_current = np.matmul(Rorbslam,ref)
        camera_prev = np.matmul(camera_R_prev,ref)
        camera_current = np.matmul(camera_R,ref)
        ang_diff_orbslam = angle_between(orbslam_prev,orbslam_current)
        ang_diff_camera = angle_between(camera_prev,camera_current)
        angle_diff = abs(math.atan2(math.sin(ang_diff_orbslam - ang_diff_camera),math.cos(ang_diff_orbslam - ang_diff_camera)))

        print('Frame: %d, -> orb: %f, cam: %f, diff: %f'%(frame,math.degrees(ang_diff_orbslam),math.degrees(ang_diff_camera),math.degrees(angle_diff)))
        row.append(math.degrees(ang_diff_orbslam))
        row.append(math.degrees(ang_diff_camera))
        row.append(math.degrees(angle_diff))
        Rorbslam_1D = np.reshape(Rorbslam,9)
        for elem in Rorbslam_1D:
            row.append(elem)
        new_csv.append(row)
        Rorbslam_prev=Rorbslam
        camera_R_prev=camera_R
    df = pd.DataFrame(np.asarray(new_csv))
    df.to_csv(os.path.join(base,'orbslam_rmatrices.csv',),header=False, index=False, sep=';', quotechar='|')
              

def main():     
        process()
      
if __name__ == "__main__": 
    main() 
