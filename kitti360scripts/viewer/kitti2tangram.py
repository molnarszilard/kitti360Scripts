
#Python code to create teh DOBB dataset from Kitti360 
# importing the required modules
import os
from kitti360scripts.helpers.annotation  import Annotation3D
import numpy as np  
import math
from kitti360scripts.helpers.project import CameraPerspective as Camera
import argparse
import json 
import shutil

np.set_printoptions(suppress=True, precision=6)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--kitti_root',default='/mnt/cuda_external_5TB/datasets/kitti/kitti360/KITTI-360/',
                    help='the root of the kitti folder of original data')
parser.add_argument('--cameraID', default='image_00',
                    help='default camera ID')
parser.add_argument('--result_folder',default='/mnt/ssd2/datasets/kitti360_visloc/',
                    help='the root folder of the results')
args = parser.parse_args()

N = 1000
### https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/helpers/labels.py
# all_classes = ['building','pole','traffic light','traffic sign','person','rider','car','truck','bus','caravan','trailer','train','motorcycle','bicycle','garage','stop','smallpole','lamp','trash bin','vending machine']
# all_classes_sID=[11,17,19,20,24,25,26,27,28,29,30,31,32,33,34,36,37,38,39,40]
# chosen_classes = ['car','rider','truck','bus','caravan','trailer','train','motorcycle','bicycle']
# chosen_classes_sID=[26,25,27,28,29,30,31,32,33]
# chosen_classes = ['car','truck','bus','caravan','trailer','train']
# chosen_classes_sID=[26,27,28,29,30,31]
# chosen_classes = ['car']
# chosen_classes_sID=[26]
chosen_classes = ['car','truck','bus','caravan','trailer','train','building']
chosen_classes_sID=[26,27,28,29,30,31,11]
# chosen_classes = all_classes

sequences = ['2013_05_28_drive_0000_sync',
             '2013_05_28_drive_0002_sync',
             '2013_05_28_drive_0003_sync',
             '2013_05_28_drive_0004_sync',
             '2013_05_28_drive_0005_sync',
             '2013_05_28_drive_0006_sync',
             '2013_05_28_drive_0007_sync',
             '2013_05_28_drive_0009_sync',
             '2013_05_28_drive_0010_sync']

if not os.path.exists(args.result_folder):
    os.makedirs(args.result_folder)

### Comparing two 3 dimensional vectors
### From https://stackoverflow.com/a/13849249
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def process():
    ids_class0 = []
    ids_class1 = []
    categories={
        "0": "vehicle",
        "1": "building",
    }
    objects = []
    nr_seq=0
    for seq in sequences:
        ids_class0 = []
        ids_class1 = []
        ### Reading 3D bounding boxes
        kitti_3dbboxes = os.path.join(args.kitti_root,'data_3d_bboxes')
        annotation3D = Annotation3D(kitti_3dbboxes, seq,list_objects=False)
          
        for u_id in annotation3D.objects:
            i_id = u_id%N
            s_id = u_id//N
            obj = annotation3D(s_id, i_id)
            if obj:
                if not obj.name in chosen_classes:
                    continue
                object_id = nr_seq*1000+i_id
                if obj.name=='building':
                    category_id=1
                    ids_class1.append(object_id)
                else:
                    category_id=0
                    ids_class0.append(object_id)
                
                Xaxis = np.array([1.0,0.0,0.0])
                # Yaxis = np.array([0.0,1.0,0.0])
                # Zaxis = np.array([0.0,0.0,1.0])
                # one_axis = np.array([1.0,1.0,1.0])
                new_Xaxis = unit_vector(obj.R@Xaxis)
                # new_Yaxis = unit_vector(obj.R@Yaxis)
                # new_Zaxis = unit_vector(obj.R@Zaxis)
                # new_one_axis = unit_vector(obj.R@one_axis)
                center = [obj.T[0],obj.T[1],obj.T[2]]
                obj_R_new = rotation_matrix_from_vectors(Xaxis,new_Xaxis)
                RorMat = [obj_R_new[0,:],obj_R_new[1,:],obj_R_new[2,:]]
                # obj_R_new2 = rotation_matrix_from_vectors(one_axis,new_one_axis)
                # oR = (np.linalg.inv(obj.R).T + obj.R)/2
                new_vertices = (obj_R_new.T)@((obj.vertices-np.expand_dims(obj.T, axis=0)).T)
                ellipseXaxis=abs(new_vertices[0].max()-new_vertices[0].min())/2
                ellipseYaxis=abs(new_vertices[1].max()-new_vertices[1].min())/2
                ellipseZaxis=abs(new_vertices[2].max()-new_vertices[2].min())/2
                axes = [ellipseXaxis,ellipseYaxis,ellipseZaxis]
                ellipsoid = {}
                ellipsoid["axes"]=axes
                ellipsoid["R"]=RorMat
                ellipsoid["center"]=center

                object = {}
                object["category_id"]=category_id
                object["object_id"]=object_id
                object["ellipsoid"]=ellipsoid
                objects.append(object)
        # print(ids_class0)
        # print(ids_class1)
        nr_seq+=1

    new_json = {}
    new_json["category_id_to_label"]=categories
    new_json["objects"]=objects

    with open(os.path.join(args.result_folder, 'visloc_kitti360_scene.json'), 'w', encoding='utf-8') as f_json:
        json.dump(new_json, f_json, indent=4, separators=(',', ': '), cls=NumpyEncoder)

    for seq in sequences:
        if not os.path.exists(os.path.join(args.result_folder,seq)):
            os.makedirs(os.path.join(args.result_folder,seq))
        kitti_image = os.path.join(args.kitti_root,'data_2d_raw',seq,args.cameraID,'data_rect')
        camera = Camera(root_dir=args.kitti_root, seq=seq) 
        imagenames=[]
        dlist=os.listdir(kitti_image)
        dlist.sort()
        for filename in dlist:
            if filename.endswith(".png"):
                imagenames.append(filename)
            else:
                continue
        if len(imagenames)<1:
            print("%s is empty"%(kitti_image))
            continue
        for filename in imagenames:            
            frame = int(os.path.splitext(filename)[0])
            if frame==0:
                continue
            print("Frame: %d"%(frame))
            valid_key_found = False
            valid_key = frame
            while not valid_key_found:
                try:
                    camera_tr = camera.cam2world[valid_key]
                    valid_key_found=True
                except:
                    valid_key-=1
            new_imgname = 'frame-'+f'{frame:06d}'+".color.png"
            new_txtname = 'frame-'+f'{frame:06d}'+".pose.txt"
            shutil.copy2(os.path.join(args.kitti_root,'data_2d_raw',seq,args.cameraID,'data_rect',filename),os.path.join(args.result_folder,seq,new_imgname))
            np.savetxt(os.path.join(args.result_folder,seq,new_txtname),camera_tr,fmt='%.8f')

def main():     
    process()
      
if __name__ == "__main__": 
    main() 
