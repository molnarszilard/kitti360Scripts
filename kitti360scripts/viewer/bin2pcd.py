
#Python code to illustrate parsing of XML files 
# importing the required modules 
import os
import numpy as np
import open3d as o3d
import struct

kitti_root = '/mnt/cuda_external_5TB/datasets/kitti/kitti360/KITTI-360/'
sequence = '2013_05_28_drive_0010_sync'

name = '0000000120.bin'
sensor_type = 'velodyne_points' # 'sick_points'
kitti_bbox = os.path.join(kitti_root,'data_3d_raw/',sequence,sensor_type,'data',name)


### Creating Output structure
base=os.path.join('/mnt/ssd2/datasets/kitti360_3d/',sequence)

if not os.path.exists(base):
    os.makedirs(base)

def main():     
    
    # Load binary point cloud
    bin_pcd = np.fromfile(kitti_bbox, dtype=np.float32)

    # Reshape and drop reflection values
    points = bin_pcd.reshape((-1, 4))[:, 0:3]

    # Convert to Open3D point cloud
    o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

    # Save to whatever format you like
    o3d.io.write_point_cloud(os.path.join(base,name[:-4])+".pcd", o3d_pcd)
      
if __name__ == "__main__": 
  
    # calling main function 
    main() 
