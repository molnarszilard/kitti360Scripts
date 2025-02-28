
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

np.set_printoptions(suppress=True, precision=6)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--kitti_root',default='/mnt/cuda_external_5TB/datasets/kitti/kitti360/KITTI-360/',
                    help='the root of the kitti folder of original data')
parser.add_argument('--sequence', default='2013_05_28_drive_0010_sync',
                    help='the sequence')
parser.add_argument('--list', default='/mnt/ssd2/datasets/kitti360_pose2_veh_build/2013_05_28_drive_0010_sync/frames_class_both.txt',  #'/mnt/ssd2/datasets/kitti360_pose2_veh_build/2013_05_28_drive_0010_sync/frames_class_both.txt'
                    help='path to a txt file, which contains the frames that should be considered in the comparison')
parser.add_argument('--cameraID', default='image_00',
                    help='default camera ID')
parser.add_argument('--result_folder',default='/mnt/ssd2/datasets/kitti360_temp/',
                    help='the root folder of the results')
args = parser.parse_args()

camera = Camera(root_dir=args.kitti_root, seq=args.sequence)

### Creating Output structure
base=os.path.join(args.result_folder,args.sequence)
colmap_results_path=os.path.join(args.kitti_root,'colmap_results',args.sequence+'.txt')
new_csv_path='cm_rmatrices'

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

def process():
    f_label = open(colmap_results_path, "r")
    Lines_colmap = f_label.readlines()
    f_label.close()
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
        frames2check=None

    colmap_results=[]
    line_number=-1
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
        if camera_ID!=1:
            print("Camera is not 1, but %d"%(camera_ID))
        frame = int(os.path.splitext(elements[9])[0])
        colmap_results.append([frame,qw,qx,qy,qz,tx,ty,tz])
        
    # colmap_results=np.asarray(colmap_results)
    colmap_results=sorted(colmap_results, key=lambda x: x[0])
    colmap_results=np.asarray(colmap_results)
    new_csv = []
    row=[]
    row.append('ImageName')
    row.append('angle_error')
    row.append('trans_error')
    row.append('Rc2w-11')
    row.append('Rc2w-12')
    row.append('Rc2w-13')
    row.append('Rc2w-21')
    row.append('Rc2w-22')
    row.append('Rc2w-23')
    row.append('Rc2w-31')
    row.append('Rc2w-32')
    row.append('Rc2w-33')
    row.append('Tc2w-x')
    row.append('Tc2w-y')
    row.append('Tc2w-z')
    new_csv.append(row)
    first_frame=True
    for colmap_estimate in colmap_results:
        frame,qw,qx,qy,qz,tx,ty,tz=colmap_estimate
        frame=int(frame)
        if frame==0 :# I do not know what to do with image 0000000000.png as it is not annotated
            continue
        if frames2check is not None and not frame in frames2check:
            continue
        # if frame>150:
        #     continue
        # print(frame)
        row = []
        row.append(f'{frame:010d}.png')
        qvec = [qw,qx,qy,qz]
        Rcolmap=qvec2rotmat(qvec)
        Tcolmap=np.array([tx,ty,tz])
        

        frame=int(frame)
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
        print("")
        print(camera_T)

        if first_frame:
            first_frame=False
            rotZ_init = np.asarray(Rotation.from_euler('Z', 90, degrees=True).as_matrix())
            rotY_init = np.asarray(Rotation.from_euler('Y', -90, degrees=True).as_matrix())
            Rcolmap_init1 = np.matmul(rotZ_init,rotY_init)
            Rcolmap_new=np.matmul(Rcolmap,Rcolmap_init1)
            ref=np.array([1.0,0.0,0.0])
            colmapX = np.matmul(Rcolmap_new.T,ref)            
            cameraX = np.matmul(camera_R,ref)
            # ang_diff = angle_between(colmapX,cameraX)
            angle_colmap = math.atan2(colmapX[1],colmapX[0])
            angle_camera = math.atan2(cameraX[1],cameraX[0])
            ang_diff = math.atan2(math.sin(angle_camera - angle_colmap),math.cos(angle_camera - angle_colmap))
            rotY_init2 = np.asarray(Rotation.from_euler('Y', -ang_diff, degrees=False).as_matrix())
            # Rcolmap_new=np.matmul(Rcolmap_new.T,rotY_init2)
            # Rcolmap_init=np.matmul(np.matmul(rotZ_init,rotY_init).T,rotY_init2)
            Rcolmap_new=np.matmul(np.matmul(Rcolmap,Rcolmap_init1).T,rotY_init2)
            print(Tcolmap)
            print(-Rcolmap.T@Tcolmap)
            # print(-Rcolmap_new.T@Tcolmap)
            Tcm_min=min(-Tcolmap)
            Tcm_max=max(-Tcolmap)
            Tcam_min=min(camera_T)
            Tcam_max=max(camera_T)
            Tcm_init = (-Tcolmap-Tcm_min)/(Tcm_max-Tcm_min)*(Tcam_max-Tcam_min)+Tcam_min
            Tcm_initZ=camera_T-Tcm_init
            Tcolmap_new=(-Tcolmap-Tcm_min)/(Tcm_max-Tcm_min)*(Tcam_max-Tcam_min)+Tcam_min+Tcm_initZ
        else:
            print(Tcolmap)
            print(-Rcolmap.T@Tcolmap)
            # print(-Rcolmap_new.T@Tcolmap)
            Rcolmap_new=np.matmul(np.matmul(Rcolmap,Rcolmap_init1).T,rotY_init2)
            Tcolmap_new=(-Tcolmap-Tcm_min)/(Tcm_max-Tcm_min)*(Tcam_max-Tcam_min)+Tcam_min+Tcm_initZ
            
        
        # rotX90 = np.asarray(Rotation.from_euler('X', 90, degrees=True).as_matrix())
        # # rotZ90 = np.asarray(Rotation.from_euler('Z', 90, degrees=True).as_matrix())
        # unitCY=np.array([0.0,0.0,1.0])
        # ref=np.array([1.0,0.0,0.0])
        # cam2world2 = np.matmul(camera_R,rotX90)
        # camRzi,camRyi,camRxi = geometry_utils.decompose_camera_rotation(np.linalg.inv(cam2world2),order='ZYX')
        # rotYwimu = np.asarray(Rotation.from_euler('Y', camRyi, degrees=True).as_matrix())
        # rotXwimu = np.asarray(Rotation.from_euler('X', camRxi, degrees=True).as_matrix())
        # rotIMU = np.matmul(rotYwimu,rotXwimu)
        # cam2world_norm = np.matmul(rotIMU,cam2world2)

        # r0 = Rotation.identity()
        # ax = plt.figure().add_subplot(projection="3d", proj_type="ortho")
        # plot_rotated_axes(ax, r0, name="W", offset=(0, 0, 0))                    
        # plot_rotated_axes(ax, Rotation.from_matrix(camera_R), name="cR", offset=(2, 0, 0))
        # plot_rotated_axes(ax, Rotation.from_matrix(camera_R.T), name="cRt", offset=(4, 0, 0))
        # # plot_rotated_axes(ax, Rotation.from_matrix(cam2world2), name="c1", offset=(4, 0, 0))
        # # plot_rotated_axes(ax, Rotation.from_matrix(cam2world_norm), name="c2", offset=(6, 0, 0))
        # plot_rotated_axes(ax, Rotation.from_matrix(Rcolmap), name="rc", offset=(6, 0, 0))
        # plot_rotated_axes(ax, Rotation.from_matrix(Rcolmap.T), name="rct", offset=(8, 0, 0))
        # plot_rotated_axes(ax, Rotation.from_matrix(Rcolmap_new), name="rc", offset=(10, 0, 0))
        # plot_rotated_axes(ax, Rotation.from_matrix(Rcolmap_new.T), name="rct", offset=(12, 0, 0))
        # # plot_rotated_axes(ax, Rotation.from_matrix(Rcolmap2), name="rc", offset=(10, 0, 0))
        # # plot_rotated_axes(ax, Rotation.from_matrix(Rcolmap2.T), name="rct", offset=(20, 0, 0))
        # # ax.plot([6.0,6.0], [0.0,0.0], [0.0,-1.0], "#a400ff")
        # ax.set(xlim=(-1.25, 1.25), ylim=(-1.25, 1.25), zlim=(-1.25, 1.25))
        # ax.set(xticks=range(-1, 14), yticks=[-1, 0, 1], zticks=[-1, 0, 1])
        # ax.set_aspect("equal", adjustable="box")
        # ax.figure.set_size_inches(6, 5)
        # plt.tight_layout()
        # plt.show()

        ref=np.array([1.0,0.0,0.0])
        colmapX = np.matmul(Rcolmap_new,ref)
        cameraX = np.matmul(camera_R,ref)
        ang_diff = angle_between(colmapX,cameraX)
        # angle_colmap = math.atan2(colmapX[1],colmapX[0])
        # angle_camera = math.atan2(cameraX[1],cameraX[0])
        # ang_diff = abs(math.atan2(math.sin(angle_camera - angle_colmap),math.cos(angle_camera - angle_colmap))) 
        squared_dist = np.sum((camera_T-Tcolmap_new)**2, axis=0)
        dist = np.sqrt(squared_dist)
        # print('Frame: %d, -> %f      ->   %f'%(frame,math.degrees(ang_diff),dist))
        row.append(math.degrees(ang_diff))
        row.append(dist)
        Rcolmap_1D = np.reshape(Rcolmap_new,9)
        for elem in Rcolmap_1D:
            row.append(elem)                
        row.append(Tcolmap_new[0])
        row.append(Tcolmap_new[1])
        row.append(Tcolmap_new[2])
        new_csv.append(row)
    df = pd.DataFrame(np.asarray(new_csv))
    df.to_csv(os.path.join(base,'colmap_rmatrices.csv',),header=False, index=False, sep=';', quotechar='|')
              

def main():     
        process()
      
if __name__ == "__main__": 
    main() 
