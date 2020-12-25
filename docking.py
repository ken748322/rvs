import open3d as o3d
import func
import ransac
import numpy as np


def docking(target, source):
    """
    input:
    target&source's type is func.init_pcd
    """

    # compute feature 
    voxel_size = 0.8

    source.down_sample(voxel_size)
    target.down_sample(voxel_size)

    source.change_all_color(color="blue", which_pcd=2)
    target.change_all_color(color="yellow", which_pcd=2)

    target.estimate_normal(voxel_size * 2, 30, True)
    target.invert_normal()
    source.estimate_normal(voxel_size * 2, 30, True)

    target.calculate_fpfh(voxel_size * 8, 750)
    source.calculate_fpfh(voxel_size * 8, 750)

    # top-n fpfh matching
    corr = np.array([], dtype=np.int).reshape(0, 2)

    for source_idx in range(len(source.pcd.points)):
        corr_indexs, _ = func.one_point_matching(source, target, source_idx)
        corr = np.append(corr, corr_indexs, axis=0)

    corr = o3d.utility.Vector2iVector(corr) 

    # registration
    criteria = ransac.RANSACConvergenceCriteria(3000000, 0.999)
    result = ransac.RegistrationRANSACBasedOnCorrespondence(source.pcd, target.pcd, corr, 1.5, 4, criteria)

    # global registration done 
    source.transform(result.transformation)

    # local registration
    result = func.icp(source, target, 1.5)
    source.transform(result.transformation)

    return result
