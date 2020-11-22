"""
code for open3d 0.10.0
"""

import open3d as o3d
import numpy as np
import copy


class init_pcd:
    def __init__(self, file_pass):
        self.pcd = o3d.io.read_point_cloud(file_pass)
        self.pcd_full_points = copy.deepcopy(self.pcd)

    def down_sample(self, voxel_size):
        self.pcd = self.pcd.voxel_down_sample(voxel_size)

    def estimate_normal(self, radius, max_nn, reestimate=False):
        if self.pcd.has_normals() and reestimate==False:
            print(":: Already have normal")
        else:
            # print(":: Estimate normal with search radius %.3f." % radius)
            # print("::                      max_nn  %d." % max_nn)
            self.pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))

    def calculate_fpfh(self, radius, max_nn):
        # print(":: Compute FPFH feature with search radius %.3f." % radius)
        # print("::                           max_nn  %d." % max_nn)
        self.pcd_fpfh = o3d.registration.compute_fpfh_feature(
            self.pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))

    def change_all_color(self, color='blue', which_pcd=2):

        COLORS = {'blue': [0, 0.651, 0.929],
                  'yellow': [1, 0.706, 0],
                  'pink': [0.91, 0.65, 0.82],
                  'gray': [0.68, 0.68, 0.68]}
    
        if which_pcd == 0:
            self.pcd.paint_uniform_color(COLORS[color])
        elif which_pcd == 1:
            self.pcd_full_points.paint_uniform_color(COLORS[color])
        elif which_pcd == 2:
            self.pcd.paint_uniform_color(COLORS[color])
            self.pcd_full_points.paint_uniform_color(COLORS[color])
        else: 
            pass 
    
    def reset_down_sample(self):
        self.pcd = copy.deepcopy(self.pcd_full_points)
    
    def invert_normal(self):
        """
        invert normal vector
        """
        self.pcd.normals = o3d.utility.Vector3dVector(np.asarray(self.pcd.normals)*(-1))
    
    def change_selected_points_color(self, index, color='pink'):

        COLORS = {'blue': [0, 0.651, 0.929],
                  'yellow': [1, 0.706, 0],
                  'pink': [0.91, 0.65, 0.82],
                  'gray': [0.68, 0.68, 0.68]}
        
        self.pcd.colors[index] = COLORS[color]

    def transform(self, trans_matrix):
        self.pcd.transform(trans_matrix)
        self.pcd_full_points.transform(trans_matrix)


def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh,
                                distance_threshold, transformation_type, n_ransac,
                                similarity_threshold, max_iter, max_valid):
    FUNCTIONS = {'PointToPlane': o3d.registration.TransformationEstimationPointToPlane,
                 'PointToPoint': o3d.registration.TransformationEstimationPointToPoint}

    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        max_correspondence_distance=distance_threshold,
        estimation_method=FUNCTIONS[transformation_type](),
        ransac_n=n_ransac,
        checkers=[o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(similarity_threshold),
                  o3d.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        criteria=o3d.registration.RANSACConvergenceCriteria(max_iter, max_valid))
    return result


def icp(source, target, threshold, transformation_type='PointToPoint', trans_init=np.identity(4)):
    """
    input: source,target's type is func.init_pcd
    """
    FUNCTIONS = {'PointToPlane': o3d.registration.TransformationEstimationPointToPlane,
                 'PointToPoint': o3d.registration.TransformationEstimationPointToPoint}

    result = o3d.registration.registration_icp(source.pcd, target.pcd, threshold, trans_init,
                                               FUNCTIONS[transformation_type]())

    return result

# source, target: open3d.geometrics.PointClouds
# source_fpfh, target_fpfh: open3d.registration.Feature
def fpfh_matching_top_n(source, target, source_fpfh, target_fpfh, source_index, topN):
    """
    Return:
        points: np.array
        lines: np.array
        idx: list
            target(pocket)'s index of matched points 
    """

    # make KDtree
    tree = o3d.geometry.KDTreeFlann(target_fpfh)
    [_, idx, _] = tree.search_knn_vector_xd(source_fpfh.data[:,source_index], topN)
    
    # points[0] is query
    points = [source.points[source_index]]
    
    # points[1:] are matched points in target 
    for i in idx:
        points.append(target.points[i])
    points = np.array(points)
    
    # lines have correspondence, 
    # lines: (source_index, target_index)
    # lines = np.array([[0 for i in idx], [j+1 for j in range(len(idx))]]).T
    
    return points, list(idx) 

# points: np.array
# lines: np.array
def set_lines(points):
    """
    Input: np.array
        vertex of lines,
        points[0] is query
        points[1:] are matched points in target 
    Return:
        line_set: open3d.geometry.LineSet
    """

    n = len(points)
    lines = np.array([[0 for i in range(1, n)], [j for j in range(1, n)]]).T
    
    line_set = o3d.geometry.LineSet(
        points = o3d.utility.Vector3dVector(points),
        lines = o3d.utility.Vector2iVector(lines)
    )

    line_set.paint_uniform_color([1,0,0])

    return line_set


def matched_feature_clustering(points):
    """
    Input:
        points: numpy.array
    Return:
        clustered_points: numpy.array
            clustered_points[0] -> リガンドのクエリ
            clustered_points[1:] -> ポケットの対応点
        index: numpy.array
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[1:])

    l = list(pcd.cluster_dbscan(eps=2.0, min_points=1))
    index = [l.index(i) for i in set(list(l))]

    selected_points = np.asarray(pcd.points)[index]
    clustered_points = points[0].reshape(1,3)
    clustered_points = np.append(clustered_points, selected_points, axis=0)
    
    return clustered_points, index


def one_point_matching(source, target, source_idx):
    """
    inputs are func.init_pcd
    """
    matched_points, matched_index = fpfh_matching_top_n(source.pcd, target.pcd, source.pcd_fpfh, target.pcd_fpfh, source_idx, 20)

    # line_set = set_lines(matched_points)

    clustered_points, clustered_index = matched_feature_clustering(matched_points)

    # clustered_lines_set = set_lines(clustered_points)

    target_idxs = np.array(matched_index)[clustered_index]
    corr_indexs = np.array([np.full(len(target_idxs), source_idx), target_idxs])

    return corr_indexs.T, clustered_points


