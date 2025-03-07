
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
parser.add_argument('--sequence', default='2013_05_28_drive_0009_sync',
                    help='the sequence')
parser.add_argument('--cameraID', default='image_00',
                    help='default camera ID')
parser.add_argument('--multiprocess', default=1,type=int,
                    help='number of parallell processes')
parser.add_argument('--result_folder',default='/mnt/ssd2/datasets/kitti360_pose3_veh_build/',
                    help='the root folder of the results')
args = parser.parse_args()

N = 1000
all_classes = ['building','pole','traffic light','traffic sign','person','rider','car','truck','bus','caravan','trailer','train','motorcycle','bicycle','garage','stop','smallpole','lamp','trash bin','vending machine']
# chosen_classes = ['car','rider','truck','bus','caravan','trailer','train','motorcycle','bicycle']
# chosen_classes = ['car','truck','bus','caravan','trailer','train']
chosen_classes = ['car','truck','bus','caravan','trailer','train','building']
# chosen_classes = ['car']
# chosen_classes = all_classes

kitti_image = os.path.join(args.kitti_root,'data_2d_raw',args.sequence,args.cameraID,'data_rect')
### Creating Output structure
base=os.path.join(args.result_folder,args.sequence)

csv_labels_folder=os.path.join(args.cameraID,'ellipse_dir_data_pred/')

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

def process(filenames):
    rows = []
    new_csv_path = os.path.join(base,args.sequence+"_most_commoon_objects.txt")        
    new_csv = []
    unique_ids = []
    every_id = []
    every_frame = []
    for filename in filenames:
        frame = int(os.path.splitext(filename)[0])
        print("Frame: %d"%(frame))
        with open(os.path.join(base,csv_labels_folder, filename), newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=';', quotechar='|')
            counter = -1
            this_row=[]
            for row in csvreader:                
                counter+=1
                if counter==0:                    
                    this_row.append(frame)
                    continue                
                i_id = int(float(row[4]))
                s_id = int(float(row[3]))
                this_id = s_id*1000+i_id
                every_id.append(this_id)
                every_frame.append(frame)                
                if not this_id in unique_ids:
                    unique_ids.append(this_id)
                this_row.append(s_id*1000+i_id)
            rows.append(this_row)
    # occurences = np.zeros_like(unique_ids)


    occurences=[]
    fl = open(new_csv_path, "w")
    for i in range(len(unique_ids)):
        this_occurences = []
        this_occurences.append(unique_ids[i])
        indices = np.where(np.asarray(every_id) == unique_ids[i])[0]
        this_occurences.append(len(indices))
        fl.write("%d %d %d "%(int(unique_ids[i]/1000),unique_ids[i]%1000,len(indices)))
        for ind in indices:
            this_occurences.append(every_frame[ind])
            fl.write("%d "%(every_frame[ind]))
        occurences.append(this_occurences)
        fl.write("\n")
    fl.close()



    # df = pd.DataFrame(np.asarray(occurences))
    # df.to_csv(new_csv_path,header=False, index=False, sep=';', quotechar='|')


def main():     
    process(csv_labels)
      
if __name__ == "__main__": 
    main() 
