
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

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--kitti_root',default='/mnt/cuda_external_5TB/datasets/kitti/kitti360/KITTI-360/',
                    help='the root of the kitti folder of original data')
parser.add_argument('--sequence', default='2013_05_28_drive_0000_sync',
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
# new_images='images/'
new_labels='labels/'
if not os.path.exists(os.path.join(base,new_labels)):
    os.makedirs(os.path.join(base,new_labels))
# if not os.path.exists(os.path.join(base,new_images)):
#     os.makedirs(os.path.join(base,new_images))

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
        print("Frame: %d"%(frame))
        if frame%10:
            continue
        ### Read the camera Transformation matrix per the current frame
        ### camera_tr --> Tr(cam_0 -> world)
        camera_tr = camera.cam2world[frame]
        camera_R = camera_tr[:3, :3]
        ### THe K matrix of the camera
        camera_K = camera.K
        camera_height = camera.height
        camera_width = camera.width
        ### The camera rotation along the Z(up) axis in the worl coordinate system
        camera_angleZ = -math.atan2(camera_R[1,0],camera_R[0,0])
        img_instance = cv2.imread(os.path.join(kitti_instances,instance_name), -1)
        ### Save the new annotation file
        new_label_path = os.path.join(base,new_labels,instance_name[:-3]+'txt')
        fl = open(new_label_path, "w")
        ### The first line represents the Rotation matrix (9 values) and then the translation (last 3 values) of the camera2world
        fl.write("%f %f %f %f %f %f %f %f %f %f %f %f\n"%(camera_R[0,0],camera_R[0,1],camera_R[0,2],camera_R[1,0],camera_R[1,1],camera_R[1,2],camera_R[2,0],camera_R[2,1],camera_R[2,2],camera_tr[0,3],camera_tr[1,3],camera_tr[2,3]))
        ### THe  
        ### get the unique IDs from the instance segmentation mask     
        uniqueIDs = np.unique(img_instance)
        ### Verify every object in the current frame
        for u_id in uniqueIDs:
            i_id = u_id%N
            s_id = u_id//N
            obj = annotation3D(s_id, i_id, frame)
            if obj:
                if not obj.name in chosen_classes:
                    continue
                ### Calculate the direction
                objX1w=np.matmul(obj.R,np.array([1.0,0.0,0.0]))
                objX1c=np.matmul(np.linalg.inv(camera_R),objX1w)
                angle_dpt = -math.atan2(objX1c[2],objX1c[0]) 
                 
                vertices=obj.vertices
                ### The 3D BB has 8 points described by vertices: obj.vertices, which are in the world coordinate frame
                ### Save the vertices and the direction angle into a txt file. Every frame has a separate file, every object has a separate line
                fl.write("%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n"%(vertices[0,0],vertices[0,1],vertices[0,2],vertices[1,0],vertices[1,1],vertices[1,2],vertices[2,0],vertices[2,1],vertices[2,2],vertices[3,0],vertices[3,1],vertices[3,2],vertices[4,0],vertices[4,1],vertices[4,2],vertices[5,0],vertices[5,1],vertices[5,2],vertices[6,0],vertices[6,1],vertices[6,2],vertices[7,0],vertices[7,1],vertices[7,2],angle_dpt))
        fl.close()

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
