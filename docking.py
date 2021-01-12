import open3d as o3d
import func
import json
import numpy as np
from random import sample

data_list_file = "dud38_list.json"
data_dir_pass = "../../data/dud38_ply/"

def docking(target, source):
    """
    input:
    target&source's type is func.init_pcd
    """

    # compute feature 
    voxel_size = 0.1

    source.down_sample(voxel_size)
    target.down_sample(voxel_size)

    source.change_all_color(color="blue", which_pcd=2)
    target.change_all_color(color="yellow", which_pcd=2)

    target.estimate_normal(3.1, 30, True)
    target.invert_normal()
    source.estimate_normal(3.1, 30, True)

    target.calculate_fpfh(3.1, 135)
    source.calculate_fpfh(3.1, 135)

    # registration
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source.pcd, 
        target.pcd,
        source.pcd_fpfh,
        target.pcd_fpfh,
        1.5, 
        o3d.registration.TransformationEstimationPointToPoint(),
        4,
        [o3d.registration.CorrespondenceCheckerBasedOnEdgeLength()],
        o3d.registration.RANSACConvergenceCriteria(4000000, 500)) 
    
    # global registration done 
    source.transform(result.transformation)

    # local registration
    result = func.icp(source, target, 3.0)
    source.transform(result.transformation)

    return result

    


if __name__ == "__main__":

    # prepare data set
    with open(data_list_file, "r") as f:
        data_list = json.load(f)
    
    

   

    target = func.init_pcd(data_dir_pass + data_list["pdb_name"][3] + ".ply")
    trans_init = [[1.0, 0.0, 0.0, 15.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]]
    target.transform(trans_init)
    source = func.init_pcd(data_dir_pass + data_list["ligand_name"][3] + ".ply")
    
    # visualization 
    o3d.visualization.draw_geometries([source.pcd_full_points, target.pcd_full_points])


    # docking
    result = docking(target, source)
    print(np.asarray(result.correspondence_set).shape[0]/len(source.pcd.points))
    print()

    # visualization 
    o3d.visualization.draw_geometries([source.pcd_full_points, target.pcd_full_points])
    
    """
    # save pose
    address = "result/target3/"+"ligand"+str(i)+".ply"
    two_pcd = source.pcd_full_points + target.pcd_full_points
    o3d.io.write_point_cloud(address, two_pcd)
    """