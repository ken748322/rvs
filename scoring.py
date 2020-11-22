import open3d as o3d
import numpy as np
import copy
import func
import json 


def scoring(source, target):
    """compute the squared error with the nearest neighbor in the target of the source
    """
    error = np.average(source.pcd_full_points.compute_point_cloud_distance(target.pcd_full_points))
    return error


if __name__ == "__main__":

    data_list_file = "dud38_list.json"
    data_dir_pass = "../../data/dud38_ply/"

    # prepare data set
    with open(data_list_file, "r") as f:
        data_list = json.load(f)

    target = func.init_pcd(data_dir_pass + data_list["pdb_name"][1] + ".ply")
    source = func.init_pcd(data_dir_pass + data_list["ligand_name"][1] + ".ply")

    trans_init = [[1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]]

    target.transform(trans_init)

    # visualization 
    o3d.visualization.draw_geometries([source.pcd_full_points, target.pcd_full_points])

    # scoring
    print(scoring(source, target))

