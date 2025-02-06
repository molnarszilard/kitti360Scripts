
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
parser.add_argument('--result_folder',default='/mnt/ssd2/datasets/kitti360_temp10/',
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
new_csv_labels='csv_labels/'
if not os.path.exists(os.path.join(base,new_labels)):
    os.makedirs(os.path.join(base,new_labels))
if not os.path.exists(os.path.join(base,new_images)):
    os.makedirs(os.path.join(base,new_images))
if not os.path.exists(os.path.join(base,new_plots)):
    os.makedirs(os.path.join(base,new_plots))

### Reading instances masks    
csv_labels=[]
dlist=os.listdir(os.path.join(base,new_csv_labels))
dlist.sort()
for filename in dlist:
    if filename.endswith(".csv"):
        csv_labels.append(filename)
    else:
        continue
if len(csv_labels)<1:
    print("%s is empty"%(os.path.join(base,new_csv_labels)))
    exit()

### Reading 3D bounding boxes
kitti_3dbboxes = os.path.join(args.kitti_root,'data_3d_bboxes')
annotation3D = Annotation3D(kitti_3dbboxes, args.sequence)
camera = Camera(root_dir=args.kitti_root, seq=args.sequence)
[ts, poses] = loadPoses(os.path.join(args.kitti_root,'data_poses',args.sequence, 'poses.txt'))
poses = postprocessPoses(poses)
cam2velo = np.array([[0.04307104361, -0.08829286498, 0.995162929, 0.8043914418],[-0.999004371, 0.007784614041, 0.04392796942, 0.2993489574],[-0.01162548558, -0.9960641394, -0.08786966659, -0.1770225824],[0.0,0.0,0.0,1.0]])

def list_points(matrix,value):
    lines_points = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i,j]>0:
                if value is None:
                    lines_points.append([j,i])
                else:
                    lines_points.append(float(value[i,j])/255)
    return np.asarray(lines_points)

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

def process(filenames):
    for filename in filenames:
        frame = int(os.path.splitext(filename)[0])
        if frame%10:
            continue
        if frame!=120:
            continue

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
        cv2.imwrite(new_image_path,img_rgb)
        fl = open(new_label_path, "w")
        
        with open(os.path.join(base,new_csv_labels, filename), newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=';', quotechar='|')
            counter = -1
            for row in csvreader:
                counter+=1
                if counter==0:
                    continue
                

                i_id = int(float(row[3]))
                s_id = int(float(row[2]))
                obj = annotation3D(s_id, i_id, frame)
                if obj:
                    if not obj.name in chosen_classes:
                        continue

                    objX1w=np.matmul(obj.R,np.array([1.0,0.0,0.0]))
                    objX1c=np.matmul(np.linalg.inv(camera_R),objX1w)
                    angle_dpt = -math.atan2(objX1c[2],objX1c[0])

                    R = np.array([[row[4],row[5]],[row[6],row[7]]]).astype(np.float64)
                    w = float(row[8])*2
                    h = float(row[9])*2
                    cx = float(row[10])
                    cy = float(row[11])
                    # unit_vector = np.array([1.0,0.0])
                    # new_unit = np.matmul(R,unit_vector)
                    angle_obb = math.atan2(R[1,0],R[0,0])
                    # angle_dpt = -math.atan2(new_unit[1],new_unit[0])
                    print(math.degrees(angle_dpt))
                    corners = xywhr2xyxyxyxy(np.array([cx,cy,w,h,angle_obb]))
                    a=min(w,h)/2
                    dx=cx+a*math.cos(angle_dpt)
                    dy=cy+a*math.sin(angle_dpt)
                    fl.write("0 %f %f %f %f %f %f %f %f %f %f\n"%(corners[0,0]/W,corners[0,1]/H,corners[1,0]/W,corners[1,1]/H,corners[2,0]/W,corners[2,1]/H,corners[3,0]/W,corners[3,1]/H,dx/W,dy/H))

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
