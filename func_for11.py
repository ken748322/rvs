"""
code for open3d 0.11.1
"""

import open3d as o3d
import numpy as np
import copy
import math

if o3d.__version__ != '0.11.1':
    raise Exception("You must to use Open3d-0.11.1")

class init_pcd:
    def __init__(self, file_pass):
        self.pcd = o3d.io.read_point_cloud(file_pass)
        self.pcd_full_points = copy.deepcopy(self.pcd)

    def down_sample(self, voxel_size):
        self.pcd = self.pcd.voxel_down_sample(voxel_size)

    def estimate_normal(self, radius, max_nn):
        if self.pcd.has_normals():
            print(":: Already have normal")
        else:
            print(":: Estimate normal with search radius %.3f." % radius)
            print("::                      max_nn  %d." % max_nn)
            self.pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))

    def calculate_fpfh(self, radius, max_nn):
        print(":: Compute FPFH feature with search radius %.3f." % radius)
        print("::                           max_nn  %d." % max_nn)
        self.pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            self.pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100))

    def change_all_color(self, color='blue'):

        COLORS = {'blue': [0, 0.651, 0.929],
                  'yellow': [1, 0.706, 0],
                  'pink': [0.91, 0.65, 0.82],
                  'gray': [0.68, 0.68, 0.68]}

        self.pcd.paint_uniform_color(COLORS[color])
        self.pcd_full_points.paint_uniform_color(COLORS[color])



def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh,
                                distance_threshold, transformation_type, n_ransac,
                                similarity_threshold, max_iter, max_valid):
    FUNCTIONS = {'PointToPlane': o3d.pipelines.registration.TransformationEstimationPointToPlane,
                 'PointToPoint': o3d.pipelines.registration.TransformationEstimationPointToPoint}

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        max_correspondence_distance=distance_threshold,
        estimation_method=FUNCTIONS[transformation_type](),
        ransac_n=n_ransac,
        checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(similarity_threshold),
                  o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iter, max_valid))
    return result


def icp(source, target, threshold, transformation_type='PointToPoint', trans_init=np.identity(4)):
    FUNCTIONS = {'PointToPlane': o3d.pipelines.registration.TransformationEstimationPointToPlane,
                 'PointToPoint': o3d.pipelines.registration.TransformationEstimationPointToPoint}

    result = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init,
                                               FUNCTIONS[transformation_type]())

    return result


def jupyter_visualization(pcd):
    visualizer = o3d.JVisualizer()
    visualizer.add_geometry(pcd)
    visualizer.show()

