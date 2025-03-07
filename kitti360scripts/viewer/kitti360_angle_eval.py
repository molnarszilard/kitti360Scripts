
#Python code to create teh DOBB dataset from Kitti360 
# importing the required modules
import os
import cv2
import numpy as np  
import math
from multiprocessing import Process
import argparse
import csv
import pandas as pd

from kitti360scripts.helpers.annotation  import Annotation3D
from kitti360scripts.helpers.project import CameraPerspective as Camera
from kitti360scripts.devkits.convertOxtsPose.python.data import loadPoses
from kitti360scripts.devkits.convertOxtsPose.python.utils import postprocessPoses
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import geometry_utils

np.set_printoptions(suppress=True, precision=6)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--folder',default='/mnt/ssd2/datasets/kitti360_pose2_veh_build/',
                    help='the root folder')
parser.add_argument('--labels',default='labels_pred289/',
                    help='the labels folder')
parser.add_argument('--kitti_root',default='/mnt/cuda_external_5TB/datasets/kitti/kitti360/KITTI-360/',
                    help='the root of the kitti folder of original data')
parser.add_argument('--sequence', default='2013_05_28_drive_0009_sync',
                    help='the sequence')
parser.add_argument('--cameraID', default='image_00',
                    help='default camera ID')
parser.add_argument('--justimu', default=True, action='store_true',
                    help='do you want to use the IMU at the rotations?')
parser.add_argument('--nrcl', default=2,
                    help='number of classes')
args = parser.parse_args()

N = 1000
all_classes = ['building','pole','traffic light','traffic sign','person','rider','car','truck','bus','caravan','trailer','train','motorcycle','bicycle','garage','stop','smallpole','lamp','trash bin','vending machine']
# chosen_classes = ['car','rider','truck','bus','caravan','trailer','train','motorcycle','bicycle']
# chosen_classes = ['car','truck','bus','caravan','trailer','train']
chosen_classes = ['car','truck','bus','caravan','trailer','train','building']
# chosen_classes = ['car']
# chosen_classes = all_classes


### Creating Output structure
base=os.path.join(args.folder,args.sequence)
images='images/'
suspicious_images='sus_images/'
new_plots='plots'
new_class_lists='lists' # lists of files per classes for which image contains at least one correctly estimated object of the respective class
csv_labels_folder=os.path.join(args.cameraID,'ellipse_dir_data_gt/')
new_csv_labels_folder=os.path.join(args.cameraID,'ellipse_dir_data_pred/')

if not os.path.exists(os.path.join(base,new_plots)):
    os.makedirs(os.path.join(base,new_plots))
if not os.path.exists(os.path.join(base,suspicious_images)):
    os.makedirs(os.path.join(base,suspicious_images))
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

def _get_covariance_matrix(boxes):
    """
    Generating covariance matrix from obbs. From https://github.com/ultralytics/ultralytics

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 5) representing rotated bounding boxes, with xywhr format.

    Returns:
        (torch.Tensor): Covariance matrices corresponding to original rotated bounding boxes.
    """
    # Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.
    a, b = boxes[2:4]**2 / 12
    c = boxes[4:]
    # gbbs = np.array((boxes[2:4]**2 / 12, boxes[4:]))
    # a, b, c = gbbs.split(1, dim=-1)
    cos = math.cos(c)
    sin = math.sin(c)
    cos2 = cos**2
    sin2 = sin**2
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin

def probiou(obb1, obb2, CIoU=False, eps=1e-7):
    """
    Calculate the prob IoU between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.
    From https://github.com/ultralytics/ultralytics
    Args:
        obb1 (torch.Tensor): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
        obb2 (torch.Tensor): A tensor of shape (N, 5) representing predicted obbs, with xywhr format.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): A tensor of shape (N, ) representing obb similarities.
    """
    x1 = obb1[0]
    y1 = obb1[1]
    x2 = obb2[0]
    y2 = obb2[1]
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)

    t1 = (
        ((a1 + a2) * (y1 - y2)**2 + (b1 + b2) * (x1 - x2)**2) / ((a1 + a2) * (b1 + b2) - (c1 + c2)**2 + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2)**2 + eps)) * 0.5
    t3 = math.log(
        ((a1 + a2) * (b1 + b2) - (c1 + c2)**2)
        / (4 * math.sqrt((a1 * b1 - c1**2).clip(0) * (a2 * b2 - c2**2).clip(0)) + eps)
        + eps
    ) * 0.5
    bd = (t1 + t2 + t3).clip(eps, 100.0)
    hd = math.sqrt(1.0 - math.exp(-bd) + eps)
    iou = 1 - hd
    if CIoU:  # only include the wh aspect ratio part
        w1, h1 = obb1[2:4]
        w2, h2 = obb2[2:4]
        v = (4 / math.pi**2) * (math.atan(w2 / h2) - math.atan(w1 / h1))**2
        alpha = v / (v - iou + (1 + eps))
        return iou - v * alpha  # CIoU
    return iou

def get_center(elements):
    x1 = elements[1]
    y1 = elements[2]
    x2 = elements[3]
    y2 = elements[4]
    x3 = elements[5]
    y3 = elements[6]
    x4 = elements[7]
    y4 = elements[8]
    cx = (x1+x2+x3+x4)/4
    cy = (y1+y2+y3+y4)/4
    return cx,cy

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
    frames_class0=open(os.path.join(base,'frames_class_0.txt'), "w")
    frames_class1=open(os.path.join(base,'frames_class_1.txt'), "w")
    frames_classall=open(os.path.join(base,'frames_class_all.txt'), "w")
    frames_classboth=open(os.path.join(base,'frames_class_both.txt'), "w")
    objects_class0 = 0
    objects_class1 = 0
    every_gt_dir_angle_cl0 = []
    every_pred_dir_angle_cl0 = []
    every_gt_dir_angle_cl1 = []
    every_pred_dir_angle_cl1 = []
    every_pred_phi = []
    total_angle_diffs_camera = []
    total_angle_diffs_rot = []
    total_angle_diffs_world = []
    total_angle_diffs_conversion = []
    min_errors = []
    row=[]
    row.append("ImageName")
    row.append("angle_error")
    min_errors.append(row)
    for filename in csv_labels:
        frame_matched_cl0=False
        frame_matched_cl1=False
        frame = int(os.path.splitext(filename)[0])
        # if frame%10:
        #     continue
        # if frame!=110:
        #     continue

        print("Frame: %d"%(frame))
        valid_key_found = False
        valid_key = frame
        while not valid_key_found:
            try:
                camera_tr = camera.cam2world[valid_key]
                valid_key_found=True
            except:
                valid_key-=1
        camera_R = camera_tr[:3, :3]
        # if frame>2:
        #     exit()
        img_rgb = cv2.imread(os.path.join(base,images,filename[:-3]+'png'))
        H,W = img_rgb.shape[:2]
        label_path = os.path.join(base,args.labels,filename[:-3]+'txt')
        new_plot_path = os.path.join(base,new_plots,filename[:-3]+'png')
        new_sus_images_path = os.path.join(base,suspicious_images,filename[:-3]+'png')
        new_csv_path = os.path.join(base,new_csv_labels_folder,filename)
        f_label = open(label_path, "r")
        Lines_yolo = f_label.readlines()
        f_label.close()
        label_array = []
        for line in Lines_yolo:
            current_line = line[:-1]
            elements=current_line.split(" ")
            for i in range(len(elements)):
                if i==0:
                    elements[i]=int(elements[i])
                elif i%2:
                    elements[i]=float(elements[i])*W
                else:
                    elements[i]=float(elements[i])*H
            # label_array.append(elements)
            rect_pred = np.array((elements[1:9])).astype(np.int32)
            rect_pred_xywhr = cv2.minAreaRect(rect_pred.reshape(4, 2))
            if rect_pred_xywhr[1][1]>rect_pred_xywhr[1][0]:
                rect_pred_xywhr = ((rect_pred_xywhr[0][0],rect_pred_xywhr[0][1]),(rect_pred_xywhr[1][1],rect_pred_xywhr[1][0]),rect_pred_xywhr[2]+90)
            cx,cy = get_center(elements)
            dpt_angle = math.atan2(elements[10]-cy,elements[9]-cx)
            label_array.append([elements[0],rect_pred_xywhr[0][0],rect_pred_xywhr[0][1],rect_pred_xywhr[1][0],rect_pred_xywhr[1][1],rect_pred_xywhr[2],dpt_angle])

        base_csv = []
        ellipse_csv = []
        new_csv = []
        ids = [] ### contains the semanticID and the instanceID
        with open(os.path.join(base,csv_labels_folder, filename), newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=';', quotechar='|')
            counter = -1
            for row in csvreader:
                counter+=1
                if counter==0:
                    # row.append('dpt_angle_error(deg)')
                    row[15]="theta"
                    base_csv.append(row)
                    continue
                ids.append([int(float(row[3])),int(float(row[4]))])
                # row.append(0.0)
                base_csv.append(row)
                R = np.array([[row[5],row[6]],[row[7],row[8]]]).astype(np.float64)
                # Rc2w = np.array([[row[15],row[16],row[17]],[row[18],row[19],row[20]],[row[21],row[22],row[23]]]).astype(np.float64)
                w = float(row[9])*2
                h = float(row[10])*2
                cx = float(row[11])
                cy = float(row[12])
                angle_obb = math.degrees(math.atan2(R[1,0],R[0,0]))
                # dpt_angle_gt = float(row[13])
                dpt_pointCx = float(row[13])
                dpt_pointCy = float(row[14])
                if row[2]=='building':
                    ellipse_csv.append([1,cx,cy,w,h,angle_obb,dpt_pointCx,dpt_pointCy])
                else:
                    ellipse_csv.append([0,cx,cy,w,h,angle_obb,dpt_pointCx,dpt_pointCy])
        base_csv=np.asarray(base_csv)
        new_csv.append(base_csv[0])
        ellipse_csv=np.asarray(ellipse_csv)

        IOUs = np.zeros((len(ellipse_csv),len(label_array)))
        for Ri in range(len(ellipse_csv)):
            for Pi in range(len(label_array)):
                if ellipse_csv[Ri][0]==label_array[Pi][0]:
                    center = (label_array[Pi][1],label_array[Pi][2])
                    size = (label_array[Pi][3],label_array[Pi][4])
                    rect_pred_xywhr = (center,size,label_array[Pi][5])
                    center = (ellipse_csv[Ri][1],ellipse_csv[Ri][2])
                    size = (ellipse_csv[Ri][3],ellipse_csv[Ri][4])
                    rect_ref_xywhr = (center,size,ellipse_csv[Ri][5])
                    r1 = cv2.rotatedRectangleIntersection(rect_ref_xywhr, rect_pred_xywhr)

                    if r1[0] != 0:
                        rect_ref_xywhr_ar = np.array((rect_ref_xywhr[0][0],rect_ref_xywhr[0][1],rect_ref_xywhr[1][0],rect_ref_xywhr[1][1],rect_ref_xywhr[2]))
                        rect_pred_xywhr = np.array((rect_pred_xywhr[0][0],rect_pred_xywhr[0][1],rect_pred_xywhr[1][0],rect_pred_xywhr[1][1],rect_pred_xywhr[2]))
                        rect_ref_xywhr_ar[4] = math.radians(rect_ref_xywhr_ar[4])
                        rect_pred_xywhr[4] = math.radians(rect_pred_xywhr[4])
                        iou = probiou(rect_ref_xywhr_ar,rect_pred_xywhr)
                        if iou>=0.5:
                            IOUs[Ri,Pi]=iou
        matches = np.nonzero(IOUs >= 0.5)  # IoU > threshold and classes match
        matches = np.array(matches).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[IOUs[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        
        ### refine the camera2world rotation matrix
        rotX90 = np.asarray(Rotation.from_euler('X', 90, degrees=True).as_matrix())
        # cam2world2 = np.matmul(camera_R,rotX90)
        # rotZ,rotY,rotX = geometry_utils.decompose_camera_rotation(np.linalg.inv(cam2world2),order='ZYX')
        # rotYwimu = np.asarray(Rotation.from_euler('Y', rotY, degrees=True).as_matrix())
        # rotXwimu = np.asarray(Rotation.from_euler('X', rotX, degrees=True).as_matrix())
        # rotIMU = np.matmul(rotYwimu,rotXwimu)
        # if args.justimu:
        #     cam2world_norm = np.matmul(rotIMU,cam2world2)
        # else:
        #     cam2world_norm = cam2world2

        # cam2world2 = np.matmul(camera_R,rotX90)
        # rotZ0,rotY,rotX = geometry_utils.decompose_camera_rotation(cam2world2,order='ZYX')
        # rotZwimu = np.asarray(Rotation.from_euler('Z', rotZ0, degrees=True).as_matrix())
        # rotYwimu = np.asarray(Rotation.from_euler('Y', rotY, degrees=True).as_matrix())
        # rotXwimu = np.asarray(Rotation.from_euler('X', rotX, degrees=True).as_matrix())
        # rotIMU = np.matmul(rotYwimu,rotXwimu,rotX90.T)
        # cam2world_norm = np.matmul(rotZwimu,rotIMU)

        # cam2world2 = np.matmul(camera_R,rotX90)
        rotZ,rotY,rotX = geometry_utils.decompose_camera_rotation(camera_R,order='ZYX')
        rotZwimu = np.asarray(Rotation.from_euler('Z', rotZ, degrees=True).as_matrix())
        rotYwimu = np.asarray(Rotation.from_euler('Y', rotY, degrees=True).as_matrix())
        rotXwimu = np.asarray(Rotation.from_euler('X', rotX, degrees=True).as_matrix())
        rotIMU = rotYwimu@rotXwimu
        cam2world_norm = rotZwimu@rotIMU
        every_pred_phi.append(rotZ)

        # unitX=np.array([1.0,0.0,0.0])
        # vecX1=rotIMU@unitX
        # vecX2=camera_R@unitX
        # ang=math.degrees(angle_between(vecX1,vecX2))
        # print(camera_R)
        # print(np.matmul(rotZwimu,rotIMU)@camera_R.T)
        # print(rotZ,rotY,rotX)
        # print(math.radians(rotZ),math.radians(rotY),math.radians(rotX))
        # print(vecX1,vecX2)
        # print(ang)

        
        objects_classs0_this_frame = 0
        objects_classs1_this_frame = 0
        image_rot_error = 360
        sus_img=False
        for i in range(len(matches)):
            cx,cy,w,h,angle_obb,dpt_pointCx_gt,dpt_pointCy_gt = ellipse_csv[matches[i,0]][1:]
            cxp,cyp,wp,hp,angle_obbp,dpt_angle = label_array[matches[i,1]][1:]
            if label_array[matches[i,1]][0]==0:
                frame_matched_cl0=True
                objects_classs0_this_frame +=1   
            else:
                frame_matched_cl1=True
                objects_classs1_this_frame +=1

            
            base_csv[matches[i,0]+1][9] = wp/2
            base_csv[matches[i,0]+1][10] = hp/2
            base_csv[matches[i,0]+1][11] = cxp
            base_csv[matches[i,0]+1][12] = cyp
            base_csv[matches[i,0]+1][5] = math.cos(math.radians(angle_obbp))
            base_csv[matches[i,0]+1][6] = -math.sin(math.radians(angle_obbp))
            base_csv[matches[i,0]+1][7] = math.sin(math.radians(angle_obbp))
            base_csv[matches[i,0]+1][8] = math.cos(math.radians(angle_obbp))
            

            base_csv[matches[i,0]+1][13]=math.cos(-dpt_angle)
            base_csv[matches[i,0]+1][14]=math.sin(-dpt_angle)

            # a_vect=np.array([math.cos(-dpt_angle),math.sin(-dpt_angle),0])

            dpt_angle_gt=math.atan2(dpt_pointCy_gt,dpt_pointCx_gt)
            if label_array[matches[i,1]][0]==0:
                every_gt_dir_angle_cl0.append(math.degrees(dpt_angle_gt))
                every_pred_dir_angle_cl0.append(math.degrees(-dpt_angle))
            else:
                every_gt_dir_angle_cl1.append(math.degrees(dpt_angle_gt))
                every_pred_dir_angle_cl1.append(math.degrees(-dpt_angle))
            a=min(w,h)/2
            px=cx+a*math.cos(-dpt_angle_gt)
            py=cy+a*math.sin(-dpt_angle_gt)

            a=min(wp,hp)/2
            pxp=cxp+a*math.cos(dpt_angle)
            pyp=cyp+a*math.sin(dpt_angle)

            s_id,i_id = ids[matches[i,0]]
            obj = annotation3D(s_id, i_id, frame)
            objX1w = np.matmul(obj.R,np.array([1.0,0.0,0.0]))
            objX1w = unit_vector(objX1w)
            theta = math.atan2(objX1w[1],objX1w[0])

            phiZ = dpt_angle+theta
            rotZphiZ = np.asarray(Rotation.from_euler('Z', phiZ, degrees=False).as_matrix())
            Rc2w_pred = np.matmul(rotZphiZ,rotYwimu,rotXwimu)
            Rc2w_pred_1D = np.reshape(Rc2w_pred,9)    
            base_csv[matches[i,0]+1][15:19]=[theta,math.radians(rotY),math.radians(rotX)]

            new_csv.append(base_csv[matches[i,0]+1])
            # newa = Rc2w_pred@a_vect
            # print(a_vect)
            # print(newa)
            img_rgb = cv2.ellipse(img_rgb, (int(cx),int(cy)), (int(w/2),int(h/2)), angle_obb, 0, 360, get_color('green'), 2)
            img_rgb = cv2.ellipse(img_rgb, (int(cxp),int(cyp)), (int(wp/2),int(hp/2)), angle_obbp, 0, 360, get_color('purple'), 2)
            # img_rgb = cv2.arrowedLine(img_rgb, (int(cx),int(cy)), (int(px),int(py)), get_color('green'), 2)
            # img_rgb = cv2.arrowedLine(img_rgb, (int(cxp),int(cyp)), (int(pxp),int(pyp)), get_color('purple'), 2)

            rotZ_angle_dpt_pred = np.asarray(Rotation.from_euler('Z', -dpt_angle, degrees=False).as_matrix())
            rotZ_angle_dpt_gt = np.asarray(Rotation.from_euler('Z', dpt_angle_gt, degrees=False).as_matrix())

            unitCx=np.array([1.0,0.0,0.0])
            dirVectC_pred = np.matmul(rotZ_angle_dpt_pred,unitCx)
            dirVectC_gt = np.matmul(rotZ_angle_dpt_gt,unitCx)
            dirVectW_pred = np.matmul(Rc2w_pred,dirVectC_pred)
            dirVectW_gt = np.matmul(cam2world_norm,dirVectC_gt)

            unit_c2w_pred = np.matmul(Rc2w_pred,unitCx)
            unit_c2w_gt = np.matmul(camera_R,unitCx)

            ang_diff_camera = math.degrees(angle_between(dirVectC_pred,dirVectC_gt))
            ang_diff_world = math.degrees(angle_between(dirVectW_pred,dirVectW_gt))
            if image_rot_error>ang_diff_world:
                image_rot_error=ang_diff_world

            total_angle_diffs_rot.append(math.degrees(angle_between(unit_c2w_pred,unit_c2w_gt)))
            total_angle_diffs_camera.append(ang_diff_camera)
            total_angle_diffs_world.append(ang_diff_world)
            total_angle_diffs_conversion.append(abs(ang_diff_camera-ang_diff_world))
            # base_csv[matches[i,0]+1][19]=ang_diff_camera

            if abs(ang_diff_camera)>150:
                sus_img = True

        if len(matches)!=len(ellipse_csv):
            for i in range(len(ellipse_csv)):
                if not i in matches[:,0]:
                    cx,cy,w,h,angle_obb,dpt_pointCx_gt,dpt_pointCy_gt = ellipse_csv[i][1:]
                    dpt_angle_gt=math.atan2(dpt_pointCy_gt,dpt_pointCx_gt)
                    a=min(w,h)/2
                    px=cx+a*math.cos(-dpt_angle_gt)
                    py=cy+a*math.sin(-dpt_angle_gt)
                    img_rgb = cv2.ellipse(img_rgb, (int(cx),int(cy)), (int(w/2),int(h/2)), angle_obb, 0, 360, get_color('red'), 2)
                    # img_rgb = cv2.arrowedLine(img_rgb, (int(cx),int(cy)), (int(px),int(py)), get_color('red'), 2)
        
        if len(matches)!=len(label_array):
            for i in range(1,len(label_array)):
                if not i in matches[:,1]:
                    cxp,cyp,wp,hp,angle_obbp,dpt_angle = label_array[i][1:]  
                    a=min(wp,hp)/2
                    pxp=cxp+a*math.cos(dpt_angle)
                    pyp=cyp+a*math.sin(dpt_angle)
                    img_rgb = cv2.ellipse(img_rgb, (int(cxp),int(cyp)), (int(wp/2),int(hp/2)), angle_obbp, 0, 360, get_color('orange'), 2)
                    # img_rgb = cv2.arrowedLine(img_rgb, (int(cxp),int(cyp)), (int(pxp),int(pyp)), get_color('orange'), 2)
        if sus_img:
            cv2.imwrite(new_sus_images_path,img_rgb)
        df = pd.DataFrame(np.asarray(new_csv))
        df.to_csv(new_csv_path,header=False, index=False, sep=';', quotechar='|')
        cv2.imwrite(new_plot_path,img_rgb)
        if frame_matched_cl0:
            frames_class0.write("%d\n"%(frame))
            if args.nrcl==1:
                objects_class0+=objects_classs0_this_frame
        if frame_matched_cl1:
            frames_class1.write("%d\n"%(frame))
        if frame_matched_cl1 or frame_matched_cl0:
            frames_classall.write("%d\n"%(frame))
        if frame_matched_cl1 and frame_matched_cl0:
            frames_classboth.write("%d\n"%(frame))
            row=[]
            row.append(f'{frame:010d}.png')
            row.append(image_rot_error)
            min_errors.append(row)
            if frame_matched_cl0:
                objects_class0+=objects_classs0_this_frame
            if frame_matched_cl1:
                objects_class1+=objects_classs1_this_frame
    df = pd.DataFrame(total_angle_diffs_camera)
    df.to_csv(os.path.join(base, "angle_diff_camera.csv"),header=False, index=False)

    df = pd.DataFrame(every_gt_dir_angle_cl0)
    df.to_csv(os.path.join(base, "every_gt_dir_angle_cl0.csv"),header=False, index=False)

    df = pd.DataFrame(every_pred_dir_angle_cl0)
    df.to_csv(os.path.join(base, "every_pred_dir_angle_cl0.csv"),header=False, index=False)

    df = pd.DataFrame(every_gt_dir_angle_cl1)
    df.to_csv(os.path.join(base, "every_gt_dir_angle_cl1.csv"),header=False, index=False)

    df = pd.DataFrame(every_pred_dir_angle_cl1)
    df.to_csv(os.path.join(base, "every_pred_dir_angle_cl1.csv"),header=False, index=False)

    df = pd.DataFrame(every_pred_phi)
    df.to_csv(os.path.join(base, "every_pred_phi_camera_rot.csv"),header=False, index=False)

    df = pd.DataFrame(total_angle_diffs_world)
    df.to_csv(os.path.join(base, "angle_diff_world.csv"),header=False, index=False)

    df = pd.DataFrame(total_angle_diffs_conversion)
    df.to_csv(os.path.join(base, "angle_diff_conversion.csv"),header=False, index=False)

    df = pd.DataFrame(total_angle_diffs_rot)
    df.to_csv(os.path.join(base, "angle_diff_rotations.csv"),header=False, index=False)

    df = pd.DataFrame(np.asarray(min_errors))
    df.to_csv(os.path.join(base,'minimum_errors.csv',),header=False, index=False, sep=';', quotechar='|')

    frames_class0.close()
    frames_class1.close()
    frames_classall.close()
    frames_classboth.close()
    print("Objects in the common images of class 0 (vehicle): %d"%(objects_class0))
    print("Objects in the common images of class 1 (building): %d"%(objects_class1))

# def main():     
#     if args.multiprocess==1:
#         process(csv_labels)
#     else:
#         threads = min(args.multiprocess,len(csv_labels))
#         batches = np.array_split(csv_labels,threads)
#         active_threads = []
#         for t in range(threads):
#             p = Process(target=process, name='thread%d'%(t), args=(batches[t],))
#             p.start()
      
if __name__ == "__main__": 
    process() 
