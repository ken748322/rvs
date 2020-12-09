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
    voxel_size = 0.8

    source.down_sample(voxel_size)
    target.down_sample(voxel_size)

    source.change_all_color(color="blue", which_pcd=2)
    target.change_all_color(color="yellow", which_pcd=2)

    target.estimate_normal(voxel_size*2, 30, True)
    target.invert_normal()
    source.estimate_normal(voxel_size*2, 30, True)

    target.calculate_fpfh(6.0, 750)
    source.calculate_fpfh(6.0, 750)

    # top-n fpfh matching
    corr = np.array([], dtype=np.int).reshape(0,2)
    source_search_idxs = sample(range(len(source.pcd.points)), k=(len(source.pcd.points)//100)*50)

    for source_idx in source_search_idxs:
        corr_indexs, _ = func.one_point_matching(source, target, source_idx)
        corr = np.append(corr, corr_indexs, axis=0)

    corr = o3d.utility.Vector2iVector(corr) 

    # registration
    result = o3d.registration.registration_ransac_based_on_correspondence(
        source.pcd, 
        target.pcd,
        corr, 
        2.0, # voxel_size * 4
        o3d.registration.TransformationEstimationPointToPoint(),
        4,
        o3d.registration.RANSACConvergenceCriteria(20000000, 200000)) 
    
    # global registration done 
    source.transform(result.transformation)

    # local registration
    result = func.icp(source, target, 1.5)
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
    # o3d.visualization.draw_geometries([source.pcd_full_points, target.pcd_full_points])


    # docking
    result = docking(target, source)

    # visualization 
    # o3d.visualization.draw_geometries([source.pcd_full_points, target.pcd_full_points])

    # save pose
    func.save_pose(source, target, "out/pocket3_ligand3.ply")
   