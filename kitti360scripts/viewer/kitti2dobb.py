
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
parser.add_argument('--result_folder',default='/mnt/ssd2/datasets/kitti360_temp/',
                    help='the root folder of the results')
args = parser.parse_args()

N = 1000
all_classes = ['building','pole','traffic light','traffic sign','person','rider','car','truck','bus','caravan','trailer','train','motorcycle','bicycle','garage','stop','smallpole','lamp','trash bin','vending machine']
# chosen_classes = ['car','rider','truck','bus','caravan','trailer','train','motorcycle','bicycle']
# chosen_classes = ['car','truck','bus','caravan','trailer','train']
# chosen_classes = ['car']
chosen_classes = all_classes

kitti_image = os.path.join(args.kitti_root,'data_2d_raw',args.sequence,args.cameraID,'data_rect')
kitti_instances = os.path.join(args.kitti_root,'data_2d_semantics/train',args.sequence,args.cameraID,'instance')

### Reading instances masks    
instances=[]
dlist=os.listdir(kitti_instances)
dlist.sort()
for filename in dlist:
    if filename.endswith(".png"):
        instances.append(filename)
    else:
        continue
if len(instances)<1:
    print("%s is empty"%(kitti_instances))
    exit()

### Reading 3D bounding boxes
kitti_3dbboxes = os.path.join(args.kitti_root,'data_3d_bboxes')
annotation3D = Annotation3D(kitti_3dbboxes, args.sequence)
camera = Camera(root_dir=args.kitti_root, seq=args.sequence)

### Creating Output structure
base=os.path.join(args.result_folder,args.sequence)
new_images='images/'
new_labels='labels/'
new_plots='plots'
if not os.path.exists(os.path.join(base,new_labels)):
    os.makedirs(os.path.join(base,new_labels))
if not os.path.exists(os.path.join(base,new_images)):
    os.makedirs(os.path.join(base,new_images))
if not os.path.exists(os.path.join(base,new_plots)):
    os.makedirs(os.path.join(base,new_plots))

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

def process(instances):
    for instance_name in instances:
        frame = int(os.path.splitext(instance_name)[0])
        if frame%10:
            continue
        # if frame!=1250:
        #     continue
        ### camera_tr --> Tr(cam_0 -> world)
        camera_tr = camera.cam2world[frame]
        camera_R = camera_tr[:3, :3]
        ### From https://stackoverflow.com/questions/11514063/extract-yaw-pitch-and-roll-from-a-rotationmatrix
        camera_angleZ = -math.atan2(camera_R[1,0],camera_R[0,0])#-math.pi/2
        print("Frame: %d, Cam: %f"%(frame,math.degrees(camera_angleZ)))
        img_instance = cv2.imread(os.path.join(kitti_instances,instance_name), -1)
        H,W = img_instance.shape
        if not os.path.exists(os.path.join(kitti_image,instance_name)):
            exit()
        else:
            img_rgb = cv2.imread(os.path.join(kitti_image,instance_name))
        new_label_path = os.path.join(base,new_labels,instance_name[:-3]+'txt')
        new_image_path = os.path.join(base,new_images,instance_name)
        new_plot_path = os.path.join(base,new_plots,instance_name)
        cv2.imwrite(new_image_path,img_rgb)
        fl = open(new_label_path, "w")
        
        uniqueIDs = np.unique(img_instance)
        ### Verify every object in the current frame
        for u_id in uniqueIDs:
            i_id = u_id%N
            s_id = u_id//N
            obj = annotation3D(s_id, i_id, frame)
            if obj:
                if not obj.name in chosen_classes:
                    continue

                objX1w=np.matmul(obj.R,np.array([1.0,0.0,0.0]))
                objX1c=np.matmul(np.linalg.inv(camera_R),objX1w)
                angle_dpt = -math.atan2(objX1c[2],objX1c[0])
                # print(math.degrees(angle_dpt))

                ### obj.R --> R(object -> world)
                # angle_dpt=-math.atan2(obj.R[1,0],obj.R[0,0])#-camera_angleZ

                # Tr_obj2world = np.array([[obj.R[0,0],obj.R[0,1],obj.R[0,1],obj.T[0]],[obj.R[1,0],obj.R[1,1],obj.R[1,1],obj.T[1]],[obj.R[2,0],obj.R[2,1],obj.R[2,1],obj.T[2]],[0.0,0.0,0.0,1.0]])
                # objX0w=np.matmul(Tr_obj2world,np.array([0.0,0.0,0.0,1.0]))
                # objX1w=np.matmul(Tr_obj2world,np.array([1.0,0.0,0.0,1.0]))
                # objX0c=np.matmul(np.linalg.inv(camera_tr),objX0w)
                # objX1c=np.matmul(np.linalg.inv(camera_tr),objX1w)
                # angle_dpt = -math.atan2(objX1c[2]-objX0c[2],objX1c[0]-objX0c[0])
                # print("\nAngle")
                # print(math.degrees(angle_dpt))
                # angle_diff = abs(math.atan2(math.sin((angle_dpt-camera_angleZ) - angle_dpt),math.cos((angle_dpt-camera_angleZ) - angle_dpt)))
                # print(math.degrees(angle_diff))                

                #### Wrong approach
                # R_o2c = np.matmul(np.linalg.inv(camera_R),obj.R)
                # print("from rot matrix")
                # beta=-math.atan2(-R_o2c[2,0],math.sqrt(R_o2c[0,0]**2+R_o2c[1,0]**2))
                # print(math.degrees(beta))
                # alpha=-math.atan2(R_o2c[2,1]/math.cos(beta),R_o2c[2,2]/math.cos(beta))
                # print(math.degrees(alpha))
                # gamma=-math.atan2(R_o2c[1,0]/math.cos(beta),R_o2c[0,0]/math.cos(beta))
                # print(math.degrees(gamma))
                # beta=math.atan2(-R_o2c[2,0],math.sqrt(R_o2c[2,1]**2+R_o2c[2,2]**2))
                # print(math.degrees(beta))

                mask_instance = np.zeros_like(img_instance,dtype=np.uint8)
                mask_instance[img_instance==s_id*N+i_id] = 255
                _, thresh = cv2.threshold(mask_instance, 127, 255, 0)
                contours, _ = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                min_area=0
                if len(contours)>1:
                    for c in contours:
                        area = cv2.contourArea(c)
                        if area>min_area:
                            mask_instance = np.zeros_like(img_instance,dtype=np.uint8)
                            cv2.drawContours(mask_instance, c, -1, color=255, thickness=cv2.FILLED)
                            min_area=area
                inner_points = list_points(mask_instance,None)
                if len(inner_points)==0:
                    continue
                rect = cv2.minAreaRect(inner_points)
                Cx,Cy,lambda1,lambda2,angle=rect[0][0],rect[0][1],rect[1][0],rect[1][1],rect[2]
                angle=math.radians(angle)
                if Cx is None:
                    continue
                corners = xywhr2xyxyxyxy(np.asarray([Cx,Cy,lambda1,lambda2,angle]))
                a=max(min(lambda1,lambda2)/2,20)
                dx=Cx+a*math.cos(angle_dpt)
                dy=Cy+a*math.sin(angle_dpt)
                img_rgb = cv2.ellipse(img_rgb, (int(Cx),int(Cy)), (int(lambda1/2),int(lambda2/2)), math.degrees(angle), 0, 360, get_color(chosen_classes.index(obj.name)), 2)
                img_rgb = cv2.arrowedLine(img_rgb, (int(Cx),int(Cy)), (int(dx),int(dy)), get_color(chosen_classes.index(obj.name)), 2)
                fl.write("0 %f %f %f %f %f %f %f %f %f %f\n"%(corners[0,0]/W,corners[0,1]/H,corners[1,0]/W,corners[1,1]/H,corners[2,0]/W,corners[2,1]/H,corners[3,0]/W,corners[3,1]/H,dx/W,dy/H))
        fl.close()
        cv2.imwrite(new_plot_path,img_rgb)

def main():     
    if args.multiprocess==1:
        process(instances)
    else:
        threads = min(args.multiprocess,len(instances))
        batches = np.array_split(instances,threads)
        active_threads = []
        for t in range(threads):
            p = Process(target=process, name='thread%d'%(t), args=(batches[t],))
            p.start()
      
if __name__ == "__main__": 
    main() 
