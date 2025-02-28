#!/usr/bin/env python3

import argparse
import os
import open3d as o3d
import numpy as np
import quaternion as quat
import matplotlib.pyplot as plt
import csv

INFO = "Plot in 3D the poses of the cameras from a Kapture trajectories file."

def read_trajectories(file_path):
    trajectories = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0].startswith('#'):
                continue
            timestamp, device_id, qw, qx, qy, qz, tx, ty, tz = row
            trajectories.append((float(tx), float(ty), float(tz), float(qw), float(qx), float(qy), float(qz)))
    return trajectories

class Visualizer:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

    def add_geometry(self, geometry):
        self.vis.add_geometry(geometry)

    def run(self):
        self.vis.run()
        self.vis.destroy_window()

    def add_cameras(self, trajectories, scale=1):
        for t in trajectories:
            tx, ty, tz, qw, qx, qy, qz = t
            R = quat.as_rotation_matrix(quat.quaternion(qw, qx, qy, qz))
            t = np.array([tx, ty, tz])

            # Create camera model
            cam_model = self.draw_camera(R, t, scale, color=[1,0,0])
            for geom in cam_model:
                self.vis.add_geometry(geom)

    def draw_camera(self, R, t, scale=1, color=[0.8, 0.2, 0.8]):
        """Create axis, plane and pyramid geometries in Open3D format.
        :param R: rotation matrix
        :param t: translation
        :param scale: camera model scale
        :param color: color of the image plane and pyramid lines
        :return: camera model geometries (axis, plane and pyramid)
        """

        # 4x4 transformation
        T = np.column_stack((R, t))
        T = np.vstack((T, (0, 0, 0, 1)))

        # axis
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5 * scale)
        axis.transform(T)

        w = 1280
        h = 1024

        K = np.eye(3) * 400
        K[0,2] = w/2
        K[1,2] = h/2
        K[2,2] = 1
        Kinv = np.linalg.inv(K)

        # points in camera coordinate system
        points_pixel = [
            [0, 0, 0],
            [0, 0, 1],
            [w, 0, 1],
            [0, h, 1],
            [w, h, 1],
        ]

        points = [Kinv @ p for p in points_pixel]

        # image plane
        width = abs(points[1][0]) + abs(points[3][0])
        height = abs(points[1][1]) + abs(points[3][1])
        plane = o3d.geometry.TriangleMesh.create_box(width, height, depth=1e-6)
        plane.paint_uniform_color(color)
        plane.translate([points[1][0], points[1][1], scale])
        plane.transform(T)
        # pyramid
        points_in_world = [(R @ p + t) for p in points]
        lines = [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
        ]
        colors = [color for _ in range(len(lines))]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points_in_world),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)

        # return as list in Open3D format
        return [axis, plane, line_set]

def visualize_trajectories(trajectories):
    points = np.array([(t[0], t[1], t[2]) for t in trajectories])
    lines = [[i, i + 1] for i in range(len(points) - 1)]
    colors = [[1, 0, 0] for _ in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    vis = Visualizer()
    vis.add_geometry(line_set)
    vis.add_cameras(trajectories, scale=1.0)
    vis.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    file_path = '/home/rambo/carla_workspace/camera-pose-estimation/datasets/kapture/carla/no_dist/ClearNoon/sensors/trajectories.txt'
    trajectories = read_trajectories(file_path)
    visualize_trajectories(trajectories)