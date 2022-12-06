# Partially borrowed from Nacho's lidar odometry (KISS-ICP)

from abc import ABC
import copy
from functools import partial
import os
from typing import Callable, List

import numpy as np
import open3d as o3d
from trimesh import transform_points

YELLOW = np.array([1, 0.706, 0])
RED = np.array([128, 0, 0]) / 255.0
BLACK = np.array([0, 0, 0]) / 255.0
GOLDEN = np.array([1.0, 0.843, 0.0])


class StubVisualizer(ABC):
    def __init__(self):
        pass

    def update(self, frame, target, pose):
        pass


class MapVisualizer(StubVisualizer):
    # Public Interaface ----------------------------------------------------------------------------
    def __init__(self):
        # Initialize GUI controls
        self.block_vis = True
        self.play_crun = False
        self.reset_bounding_box = True

        # Create data
        self.scan = o3d.geometry.PointCloud()
        self.frame_axis_len = 0.8
        self.frame = o3d.geometry.TriangleMesh()
        self.mesh = o3d.geometry.TriangleMesh()

        # Initialize visualizer
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self._register_key_callbacks()
        self._initialize_visualizer()

        # Visualization options
        self.render_map = True
        self.render_frame = True

        self.global_view = False
        self.view_control = self.vis.get_view_control()
        self.camera_params = self.view_control.convert_to_pinhole_camera_parameters()

    def update(self, scan, pose, mesh = None):
        self._update_geometries(scan, pose, mesh)
        while self.block_vis:
            self.vis.poll_events()
            self.vis.update_renderer()
            if self.play_crun:
                break
        self.block_vis = not self.block_vis
    
    def destroy_window(self):
        self.vis.destroy_window()
    
    def stop(self):
        self.block_vis = True
        while self.block_vis:
            self.vis.poll_events()
            self.vis.update_renderer()
            # if self.play_crun:
            #     break

    # Private Interaface ---------------------------------------------------------------------------
    def _initialize_visualizer(self):
        w_name = self.__class__.__name__
        self.vis.create_window(window_name=w_name, width=1920, height=1080)
        self.vis.add_geometry(self.scan)
        self.vis.add_geometry(self.frame)
        self.vis.add_geometry(self.mesh)
        self._set_white_background(self.vis)
        self.vis.get_render_option().point_size = 2
        self.vis.get_render_option().light_on = True
        print(100 * "*")
        print(f"{w_name} initialized. Press [SPACE] to pause/start, [N] to step, [ESC] to exit.")

    def _register_key_callback(self, keys: List, callback: Callable):
        for key in keys:
            self.vis.register_key_callback(ord(str(key)), partial(callback))

    def _register_key_callbacks(self):
        self._register_key_callback(["Ā", "Q", "\x1b"], self._quit)
        self._register_key_callback([" "], self._start_stop)
        self._register_key_callback(["V"], self._toggle_view)
        self._register_key_callback(["F"], self._toggle_frame)
        self._register_key_callback(["M"], self._toggle_map)
        # self._register_key_callback(["B"], self._set_black_background)
        # self._register_key_callback(["W"], self._set_white_background)

    def _set_black_background(self, vis):
        vis.get_render_option().background_color = [0.0, 0.0, 0.0]

    def _set_white_background(self, vis):
        vis.get_render_option().background_color = [1.0, 1.0, 1.0]

    def _quit(self, vis):
        print("Destroying Visualizer")
        vis.destroy_window()
        os._exit(0)

    def _next_frame(self, vis):
        self.block_vis = not self.block_vis

    def _start_stop(self, vis):
        self.play_crun = not self.play_crun

    def _toggle_frame(self, vis):
        self.render_frame = not self.render_frame
        return False

    def _toggle_map(self, vis):
        self.render_map = not self.render_map
        return False
    
    def _update_geometries(self, scan, pose, mesh = None):
        # Scan (toggled by "F")
        if self.render_frame:
            self.scan.points = o3d.utility.Vector3dVector(scan.points)
            self.scan.paint_uniform_color(GOLDEN)
        else:
            self.scan.points = o3d.utility.Vector3dVector()

        # Always visualize the coordinate frame
        self.frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.frame_axis_len, origin=np.zeros(3))
        self.frame = self.frame.transform(pose)
    
        # Mesh Map (toggled by "M")
        if self.render_map:
            if mesh is not None:
                self.vis.remove_geometry(self.mesh, self.reset_bounding_box)  # if comment, then we keep the previous reconstructed mesh (for the case we use local map reconstruction) 
                self.mesh = mesh
                self.vis.add_geometry(self.mesh, self.reset_bounding_box)
        else:
            self.vis.remove_geometry(self.mesh, self.reset_bounding_box) 

        self.vis.update_geometry(self.scan)
        self.vis.add_geometry(self.frame, self.reset_bounding_box)            

        if self.reset_bounding_box:
            self.vis.reset_view_point(True)
            self.reset_bounding_box = False

    def _toggle_view(self, vis):
        self.global_view = not self.global_view
        vis.update_renderer()
        vis.reset_view_point(True)
        current_camera = self.view_control.convert_to_pinhole_camera_parameters()
        if self.camera_params and not self.global_view:
            self.view_control.convert_from_pinhole_camera_parameters(self.camera_params)
        self.camera_params = current_camera
