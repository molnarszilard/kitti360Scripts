
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
from array import array

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

def find_common_elements(arr1, arr2):
    common_elements = array('i', [x for x in arr1 if x in arr2])
    return list(common_elements)

def process(filenames):
    rows = []
    new_csv_path = os.path.join(base,args.sequence+"_most_common_frames.txt")        
    new_csv = []
    unique_ids = []
    unique_frames = []
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
                if not frame in unique_frames:
                    unique_frames.append(frame)
                this_row.append(s_id*1000+i_id)
            rows.append(this_row)
    common_matrix = np.zeros((len(rows),len(rows)))
    for i in range(len(rows)):
        print(i)
        for j in range(i,len(rows)):
            if i==j:
                continue
            this_rowI = rows[i][1:]
            this_rowJ = rows[j][1:]
            common_elements = find_common_elements(this_rowI, this_rowJ)
            if len(common_elements)>0:
                common_matrix[i,j]=len(common_elements)
            
    fl = open(new_csv_path, "w")
    for i in range(len(rows)):
        for j in range(len(rows)):
            if common_matrix[i,j]>0:
                fl.write("%d %d %d\n"%(rows[i][0],rows[j][0],common_matrix[i,j]))
    
    fl.close()



    # df = pd.DataFrame(np.asarray(occurences))
    # df.to_csv(new_csv_path,header=False, index=False, sep=';', quotechar='|')


def main():     
    process(csv_labels)
      
if __name__ == "__main__": 
    main() 
