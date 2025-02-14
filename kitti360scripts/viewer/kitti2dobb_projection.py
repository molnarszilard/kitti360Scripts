
#Python code to create teh DOBB dataset from Kitti360 
# importing the required modules
import os
from kitti360scripts.helpers.annotation  import Annotation3D
import cv2
import numpy as np  
import math
from kitti360scripts.helpers.project import CameraPerspective as Camera
from multiprocessing import Process
import argparse
from kitti360scripts.devkits.convertOxtsPose.python.data import loadPoses
from kitti360scripts.devkits.convertOxtsPose.python.utils import postprocessPoses
import csv
import pandas as pd
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import geometry_utils

np.set_printoptions(suppress=True, precision=6)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--kitti_root',default='/mnt/cuda_external_5TB/datasets/kitti/kitti360/KITTI-360/',
                    help='the root of the kitti folder of original data')
parser.add_argument('--sequence', default='2013_05_28_drive_0010_sync',
                    help='the sequence')
parser.add_argument('--cameraID', default='image_00',
                    help='default camera ID')
parser.add_argument('--multiprocess', default=1,type=int,
                    help='number of parallell processes')
parser.add_argument('--result_folder',default='/mnt/ssd2/datasets/kitti360_pose/',
                    help='the root folder of the results')
args = parser.parse_args()

N = 1000
all_classes = ['building','pole','traffic light','traffic sign','person','rider','car','truck','bus','caravan','trailer','train','motorcycle','bicycle','garage','stop','smallpole','lamp','trash bin','vending machine']
# chosen_classes = ['car','rider','truck','bus','caravan','trailer','train','motorcycle','bicycle']
chosen_classes = ['car','truck','bus','caravan','trailer','train']
# chosen_classes = ['car']
# chosen_classes = all_classes

kitti_image = os.path.join(args.kitti_root,'data_2d_raw',args.sequence,args.cameraID,'data_rect')
### Creating Output structure
base=os.path.join(args.result_folder,args.sequence)
new_images='images/'
new_plots='plots'
new_labels='labels/'
csv_labels_folder=os.path.join(args.cameraID,'ellipse_data/')
new_csv_labels_folder=os.path.join(args.cameraID,'ellipse_dir_data_gt/')
if not os.path.exists(os.path.join(base,new_labels)):
    os.makedirs(os.path.join(base,new_labels))
if not os.path.exists(os.path.join(base,new_images)):
    os.makedirs(os.path.join(base,new_images))
if not os.path.exists(os.path.join(base,new_plots)):
    os.makedirs(os.path.join(base,new_plots))
if not os.path.exists(os.path.join(base,new_csv_labels_folder)):
    os.makedirs(os.path.join(base,new_csv_labels_folder))

### Reading instances masks
csv_labels=[]
dlist=os.listdir(os.path.join(base,csv_labels_folder))
dlist.sort()
for filename in dlist:
    if filename.endswith(".csv"):
        csv_labels.append(filename)
    else:
        continue
if len(csv_labels)<1:
    print("%s is empty"%(os.path.join(base,csv_labels_folder)))
    exit()

### Reading 3D bounding boxes
kitti_3dbboxes = os.path.join(args.kitti_root,'data_3d_bboxes')
annotation3D = Annotation3D(kitti_3dbboxes, args.sequence)
camera = Camera(root_dir=args.kitti_root, seq=args.sequence)
[ts, poses] = loadPoses(os.path.join(args.kitti_root,'data_poses',args.sequence, 'poses.txt'))
poses = postprocessPoses(poses)
cam2velo = np.array([[0.04307104361, -0.08829286498, 0.995162929, 0.8043914418],[-0.999004371, 0.007784614041, 0.04392796942, 0.2993489574],[-0.01162548558, -0.9960641394, -0.08786966659, -0.1770225824],[0.0,0.0,0.0,1.0]])
cam2imu = np.array([[0.0371783278,-0.0986182135,0.9944306009,1.5752681039],[0.9992675562,-0.0053553387,-0.0378902567,0.0043914093],[0.0090621821,0.9951109327,0.0983468786,-0.6500000000]])

def xywhr2xyxyxyxy(rboxes):
    ctr = rboxes[:2]
    w, h, angle = rboxes[2:5]
    cos_value, sin_value = math.cos(angle), math.sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    corners = np.stack([pt1, pt2, pt3, pt4], axis=-2)
    return corners

def get_color(cls,max=10):
    if isinstance(cls,int):
        if cls%max==0: #plane
            return (0,128,255) #orange
        elif cls%max==1: #ship
            return (255,0,255) #fuchsia
        elif cls%max==2: # large vehicle
            return (255,0,0) #blue
        elif cls%max==3: # small vehicle
            return (255,255,0) #cyan
        elif cls%max==4: # helicopter
            return (255,0,128) #purple
        elif cls%max==5: # 
            return (255,153,255) # pink
        elif cls%max==6: 
            return (0,255,255) #yellow
        elif cls%max==7: 
            return (128,0,255) #magenta
        elif cls%max==8: 
            return (255,255,255) #white
        elif cls%max==9: 
            return (128,255,0) #teal
        elif cls%max==10:
            return (0,0,0) #black
        elif cls%max==11:
            return (0,255,0) #green
        elif cls%max==12:
            return (0,0,255) #red
        else:
            # print("Try another class.")
            return (0,0,0) #black
    else:
        if cls=='orange': #plane
            return (0,128,255) #orange
        elif cls=='fuchsia': #ship
            return (255,0,255) #fuchsia
        elif cls=='blue': # large vehicle
            return (255,0,0) #blue
        elif cls=='cyan': # small vehicle
            return (255,255,0) #cyan
        elif cls=='purple': # helicopter
            return (255,0,128) #purple
        elif cls=='pink':
            return (255,153,255) #pink
        elif cls=='yellow': # special 2
            return (0,255,255) #yellow
        elif cls=='magenta':
            return (128,0,255) #magenta
        elif cls=='white': # special 4
            return (255,255,255) #white
        elif cls=='teal': # container crane
            return (128,255,0) #teal
        elif cls=='black':
            return (0,0,0) #black
        elif cls=='green': # special 3
            return (0,255,0) #green        
        elif cls=='red': # special 1
            return (0,0,255) #red
        
        elif cls=='yolored':
            return (59,57,253) #yolo detection red        
        else:
            print("Try another class.")
            return (0,0,0) #black


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

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    if any(v): #if not all zeros then 
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    else:
        return np.eye(3) #cross of all zeros only occurs on identical directions


def process(filenames):
    for filename in filenames:
        frame = int(os.path.splitext(filename)[0])
        if frame%10:
            continue
        # if frame!=1250:
        #     continue

        print("Frame: %d"%(frame))
        ### camera_tr --> Tr(cam_0 -> world)
        camera_tr = camera.cam2world[frame]
        camera_R = camera_tr[:3, :3]
        camera_K = camera.K
        camera_height = camera.height
        camera_width = camera.width
        current_imu_pose = poses[np.where(ts == frame)[0][0]] 

        
        if not os.path.exists(os.path.join(kitti_image,filename[:-3]+'png')):
            exit()
        else:
            img_rgb = cv2.imread(os.path.join(kitti_image,filename[:-3]+'png'))
        H,W = img_rgb.shape[:2]
        new_label_path = os.path.join(base,new_labels,filename[:-3]+'txt')
        new_image_path = os.path.join(base,new_images,filename[:-3]+'png')
        new_plot_path = os.path.join(base,new_plots,filename[:-3]+'png')
        new_csv_path = os.path.join(base,new_csv_labels_folder,filename)
        cv2.imwrite(new_image_path,img_rgb)
        fl = open(new_label_path, "w")
        
        new_csv = []
        with open(os.path.join(base,csv_labels_folder, filename), newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=';', quotechar='|')
            counter = -1
            for row in csvreader:
                
                counter+=1
                if counter==0:
                    row.append('directionX')
                    row.append('directionY')
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
                    continue
                
                i_id = int(float(row[4]))
                s_id = int(float(row[3]))
                obj = annotation3D(s_id, i_id, frame)
                if obj:
                    if not obj.name in chosen_classes:
                        continue
                    # imu2worldR = current_imu_pose[:3,:3]
                    # imuRzi,imuRyi,imuRxi = geometry_utils.decompose_camera_rotation(np.linalg.inv(imu2worldR),order='ZYX')
                    # rotYwimu = np.asarray(Rotation.from_euler('Y', imuRyi, degrees=True).as_matrix())
                    # rotXwimu = np.asarray(Rotation.from_euler('X', imuRxi, degrees=True).as_matrix())
                    # rotYwimu180 = np.asarray(Rotation.from_euler('Y', 180, degrees=True).as_matrix())
                    # rotIMU2 = np.matmul(rotYwimu180,np.matmul(rotYwimu,rotXwimu))
                    # unitCY=np.array([0.0,0.0,1.0])
                    # gravityW=np.array([0.0,0.0,-1.0])
                    # unitCYinW = np.matmul(imu2worldR,unitCY)
                    # unitCYinW_norm = np.matmul(rotIMU2,unitCYinW)
                    # ang_diff = angle_between(gravityW,unitCYinW_norm)
                    # print(gravityW,unitCYinW_norm,math.degrees(ang_diff))
                    
                    rotX90 = np.asarray(Rotation.from_euler('X', 90, degrees=True).as_matrix())
                    # rotZ90 = np.asarray(Rotation.from_euler('Z', 90, degrees=True).as_matrix())
                    unitCY=np.array([0.0,0.0,1.0])
                    ref=np.array([1.0,0.0,0.0])
                    cam2world2 = np.matmul(camera_R,rotX90)
                    camRzi,camRyi,camRxi = geometry_utils.decompose_camera_rotation(np.linalg.inv(cam2world2),order='ZYX')
                    rotYwimu = np.asarray(Rotation.from_euler('Y', camRyi, degrees=True).as_matrix())
                    rotXwimu = np.asarray(Rotation.from_euler('X', camRxi, degrees=True).as_matrix())
                    rotIMU = np.matmul(rotYwimu,rotXwimu)
                    cam2world_norm = np.matmul(rotIMU,cam2world2)
                    Rc2w_1D = np.reshape(cam2world_norm,9)
                    # gravityC=np.array([0.0,0.0,-1.0])
                    # gravityW=np.array([0.0,0.0,-1.0])
                    # unitCGinW = np.matmul(cam2world2,gravityC)
                    # ang_diff = angle_between(gravityW,unitCGinW)
                    # print(gravityW,unitCGinW,math.degrees(ang_diff))
                    # unitCGinW_norm = np.matmul(rotIMU,unitCGinW)
                    # ang_diff = angle_between(gravityW,unitCGinW_norm)
                    # print(gravityW,unitCGinW_norm,math.degrees(ang_diff))

                    objX1w=np.matmul(obj.R,np.array([1.0,0.0,0.0]))
                    objX1c=np.matmul(np.linalg.inv(cam2world_norm),objX1w)
                    objX1c = unit_vector(objX1c)
                    angle_dpt = math.atan2(objX1c[1],objX1c[0]) #alpha
                    dir_pointC = objX1c[:2]
                    rotZ_alpha = np.asarray(Rotation.from_euler('Z', angle_dpt, degrees=False).as_matrix())
                    # unitCx=np.matmul(rotZ_alpha,np.array([1.0,0.0,0.0])) 
                    # ang_diff = angle_between(objX1c,unitCx)
                    # print(objX1c,step,math.degrees(ang_diff))

                    # unitCX=np.array([1.0,0.0,0.0])
                    # objXcnew=np.matmul(rotZ_alpha,unitCX)
                    # objX1c[2]=0
                    # ang_diff = angle_between(objX1c,objXcnew)
                    # print(objX1c,objXcnew,math.degrees(ang_diff))
                    # objXctow=np.matmul(cam2world_norm,np.matmul(rotZ_alpha,unitCX))
                    # # objXctow=np.matmul(np.matmul(np.matmul(rotYz,np.matmul(rotY,rotX90)),rotY_alpha),unitCX)
                    # ang_diff = angle_between(objX1w,objXctow)
                    # print(objX1w,objXctow,math.degrees(ang_diff))
                    # if math.degrees(ang_diff)>1:
                    #     print(objX1w,objXctow,math.degrees(ang_diff))
                    #     dummy=False

                    # r0 = Rotation.identity()
                    # ax = plt.figure().add_subplot(projection="3d", proj_type="ortho")
                    # plot_rotated_axes(ax, r0, name="W", offset=(0, 0, 0))
                    # plot_rotated_axes(ax, Rotation.from_matrix(np.matmul(rotIMU,cam2world2)), name="", offset=(0, 0, 0))

                    # # plot_rotated_axes(ax, Rotation.from_matrix(camera_R), name="c_o", offset=(2, 0, 0))
                    # # plot_rotated_axes(ax, Rotation.from_matrix(cam2world2), name="cz", offset=(4, 0, 0))
                    # # plot_rotated_axes(ax, Rotation.from_matrix(np.matmul(rotIMU,cam2world2)), name="c", offset=(6, 0, 0))

                    # ### Plotting the original direction angle into the world on each coordinate system
                    # ax.plot([0.0,0.0+objX1w[0]/np.max(abs(objX1w))], [0.0,objX1w[1]/np.max(abs(objX1w))], [0.0,objX1w[2]/np.max(abs(objX1w))], "#ff00ca")
                    # # ax.plot([2.0,2.0+objX1w[0]/np.max(abs(objX1w))], [0.0,objX1w[1]/np.max(abs(objX1w))], [0.0,objX1w[2]/np.max(abs(objX1w))], "#ff00ca")
                    # # ax.plot([4.0,4.0+objX1w[0]/np.max(abs(objX1w))], [0.0,objX1w[1]/np.max(abs(objX1w))], [0.0,objX1w[2]/np.max(abs(objX1w))], "#ff00ca")
                    # # ax.plot([6.0,6.0+objX1w[0]/np.max(abs(objX1w))], [0.0,objX1w[1]/np.max(abs(objX1w))], [0.0,objX1w[2]/np.max(abs(objX1w))], "#ff00ca")
                    # # ax.plot([8.0,8.0+objX1w[0]/np.max(abs(objX1w))], [0.0,objX1w[1]/np.max(abs(objX1w))], [0.0,objX1w[2]/np.max(abs(objX1w))], "#ff00ca")

                    # # ax.plot([0.0,0.0+objXctow[0]/np.max(abs(objXctow))], [0.0,objXctow[1]/np.max(abs(objXctow))], [0.0,objXctow[2]/np.max(abs(objXctow))], "#ff8900")
                    # # ax.plot([2.0,2.0+objXctow[0]/np.max(abs(objXctow))], [0.0,objXctow[1]/np.max(abs(objXctow))], [0.0,objXctow[2]/np.max(abs(objXctow))], "#ff8900")
                    # # ax.plot([4.0,4.0+objXctow[0]/np.max(abs(objXctow))], [0.0,objXctow[1]/np.max(abs(objXctow))], [0.0,objXctow[2]/np.max(abs(objXctow))], "#ff8900")
                    # # ax.plot([6.0,6.0+objXctow[0]/np.max(abs(objXctow))], [0.0,objXctow[1]/np.max(abs(objXctow))], [0.0,objXctow[2]/np.max(abs(objXctow))], "#ff8900")
                    # # ax.plot([0.0,objXctow[0]/np.max(abs(objXctow))], [0.0,objXctow[1]/np.max(abs(objXctow))], [0.0,objXctow[2]/np.max(abs(objXctow))], "#ffa400")
                    # loc = np.array([(6, 0, 0), (6, 0, 0)])
                    # # ax.plot([6.0,6.0], [0.0,0.0], [0.0,-1.0], "#a400ff")
                    # ax.set(xlim=(-1.25, 1.25), ylim=(-1.25, 1.25), zlim=(-1.25, 1.25))
                    # ax.set(xticks=range(-1, 2), yticks=[-1, 0, 1], zticks=[-1, 0, 1])
                    # ax.set_aspect("equal", adjustable="box")
                    # ax.figure.set_size_inches(6, 5)
                    # plt.tight_layout()
                    # plt.show()   

                    # camX10w=np.matmul(camera_R,np.array([0.0,0.0,0.0]))
                    # camX0velo=np.matmul(cam2velo,np.concatenate((np.array([0.0,0.0,0.0]),np.array([1.0]))))[:3]
                    # camX1w=np.matmul(camera_R,np.array([1.0,0.0,0.0]))
                    # camX1velo=np.matmul(cam2velo,np.concatenate((np.array([1.0,0.0,0.0]),np.array([1.0]))))[:3]
                    # camX10w=np.matmul(camera_R,np.array([10.0,0.0,0.0]))
                    # camX10velo=np.matmul(cam2velo,np.concatenate((np.array([10.0,0.0,0.0]),np.array([1.0]))))[:3]
                    
                    # WX0inVelo=np.matmul(cam2velo[:3,:3],np.matmul(np.linalg.inv(camera_R),np.array([0.0,0.0,0.0])))
                    # WX1inVelo=np.matmul(cam2velo[:3,:3],np.matmul(np.linalg.inv(camera_R),np.array([1.0,0.0,0.0])))
                    # WX10inVelo=np.matmul(cam2velo[:3,:3],np.matmul(np.linalg.inv(camera_R),np.array([10.0,0.0,0.0])))
                    
                    # beta = math.radians(camRzi)
                    # beta=math.atan2(camX1w[1],camX1w[0])
                    # angle_dpt_in_world = math.atan2(objX1w[1],objX1w[0]) #theta
                    # camera_angleZ = -math.atan2(camera_R[1,0],camera_R[0,0]) #beta
                    # print(math.degrees(beta),math.degrees(angle_dpt_in_world),math.degrees(angle_dpt),math.degrees(angle_dpt_in_world-beta+angle_dpt))
                    
                    # camRz,camRy,camRx = geometry_utils.decompose_camera_rotation(camera_R,order='ZYX')
                    # print(camRz,camRy,camRx)
                    # rot = np.asarray(Rotation.from_euler('ZYX', [camRz,camRy,camRx], degrees=True).as_matrix())
                    # print(camera_R)
                    # print("")
                    # print(rot)
                    # print("")
                    # print(np.matmul(rot,np.linalg.inv(camera_R)))
                    
                  
                    # row.append(math.degrees(angle_dpt))
                    for elem in dir_pointC:
                        row.append(elem)
                    for elem in Rc2w_1D:
                        row.append(elem)
                    # row.append(Rc2w_1D)
                    new_csv.append(row)

                    R = np.array([[row[5],row[6]],[row[7],row[8]]]).astype(np.float64)
                    w = float(row[9])*2
                    h = float(row[10])*2
                    cx = float(row[11])
                    cy = float(row[12])
                    # unit_vector = np.array([1.0,0.0])
                    # new_unit = np.matmul(R,unit_vector)
                    angle_obb = math.atan2(R[1,0],R[0,0])
                    # angle_dpt = -math.atan2(new_unit[1],new_unit[0])
                    # print(math.degrees(angle_dpt))
                    corners = xywhr2xyxyxyxy(np.array([cx,cy,w,h,angle_obb]))
                    a=min(w,h)/2
                    dx=cx+a*math.cos(-angle_dpt)
                    dy=cy+a*math.sin(-angle_dpt)
                    fl.write("0 %f %f %f %f %f %f %f %f %f %f\n"%(corners[0,0]/W,corners[0,1]/H,corners[1,0]/W,corners[1,1]/H,corners[2,0]/W,corners[2,1]/H,corners[3,0]/W,corners[3,1]/H,dx/W,dy/H))
                    img_rgb = cv2.ellipse(img_rgb, (int(cx),int(cy)), (int(w/2),int(h/2)), math.degrees(angle_obb), 0, 360, get_color('red'), 2)
                    # Tr_obj2world = np.array([[obj.R[0,0],obj.R[0,1],obj.R[0,1],obj.T[0]],[obj.R[1,0],obj.R[1,1],obj.R[1,1],obj.T[1]],[obj.R[2,0],obj.R[2,1],obj.R[2,1],obj.T[2]],[0.0,0.0,0.0,1.0]])
                    new_vertices_cam = np.zeros_like(obj.vertices)
                    new_vertices_velo = np.zeros_like(obj.vertices)
                    new_projected_points = np.zeros((len(obj.vertices),2)).astype(np.int32)

                    fx = camera_K[0,0]
                    fy = camera_K[1,1]
                    x0 = camera_K[0,2]
                    y0 = camera_K[1,2]
                    for i in range(len(obj.vertices)):
                        new_vertices_cam[i]=np.matmul(np.linalg.inv(camera_tr),np.concatenate((obj.vertices[i],np.array([1.0]))))[:3]
                        z = new_vertices_cam[i,2]
                        u = (new_vertices_cam[i,0]*fx)/z
                        v = (new_vertices_cam[i,1]*fy)/z
                        pixel_pos_x = round(u + x0)
                        pixel_pos_y = round(v + y0)

                        if pixel_pos_x < 0:                    
                            pixel_pos_x = -pixel_pos_x
                        
                        if pixel_pos_x > camera_width - 1:                    
                            pixel_pos_x = camera_width - 1
                        
                        if pixel_pos_y < 0:
                            pixel_pos_y = -pixel_pos_y
                        
                        if pixel_pos_y > camera_height - 1:
                            pixel_pos_y = camera_height - 1
                        
                        new_projected_points[i,0] = pixel_pos_x
                        new_projected_points[i,1] = pixel_pos_y
                        
                        new_vertices_velo[i]=np.matmul(cam2velo,np.concatenate((new_vertices_cam[i],np.array([1.0]))))[:3]
                        # img_rgb = cv2.circle(img_rgb, (pixel_pos_x,pixel_pos_y), 1, get_color(counter), 1)
                    for i in range(len(obj.lines)):
                        img_rgb = cv2.line(img_rgb, (new_projected_points[obj.lines[i][0],0],new_projected_points[obj.lines[i][0],1]), (new_projected_points[obj.lines[i][1],0],new_projected_points[obj.lines[i][1],1]),get_color(counter), 1)
                    counter+=1

        ### Object points in the velodyne pcd: new_vertices_velo
        ### Object x vector in the velodyne pcd objX0velo,objX1velo,objX10velo
        ### Camera x vector in the velodyne pcd camX0velo,camX1velo,camX10velo
        df = pd.DataFrame(np.asarray(new_csv))
        df.to_csv(new_csv_path,header=False, index=False, sep=';', quotechar='|')
        fl.close()
        cv2.imwrite(new_plot_path,img_rgb)

def main():     
    if args.multiprocess==1:
        process(csv_labels)
    else:
        threads = min(args.multiprocess,len(csv_labels))
        batches = np.array_split(csv_labels,threads)
        active_threads = []
        for t in range(threads):
            p = Process(target=process, name='thread%d'%(t), args=(batches[t],))
            p.start()
      
if __name__ == "__main__": 
    main() 
