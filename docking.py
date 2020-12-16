import open3d as o3d
import func


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

    target.calculate_fpfh(voxel_size * 5, 750)
    source.calculate_fpfh(voxel_size * 5, 750)

    # registration
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source.pcd,
        target.pcd,
        source.pcd_fpfh,
        target.pcd_fpfh,
        1.0,
        o3d.registration.TransformationEstimationPointToPoint(),
        4,
        [o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.registration.CorrespondenceCheckerBasedOnDistance(0.2)],
        o3d.registration.RANSACConvergenceCriteria(20000000, 500000))

    # global registration done
    source.transform(result.transformation)

    # local registration
    result = func.icp(source, target, 1.5)
    source.transform(result.transformation)

    return result
