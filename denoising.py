import preprocess as pp
import open3d as o3d
import json
import numpy as np

data_list_file = "dud38_list.json"
data_dir_pass = "../../data/dud38_ply/tmp/"
save_dir = "../../data/dud38_ply/"


# remove noise from point-clouds of pocket
# input & output type: open3d.geometric.PointCloud
def pocket_denoising(pocket):
    l = pocket.cluster_dbscan(eps=0.3, min_points=5)
    t = np.argmax([l.count(i) for i in set(l)])
    return pocket.select_by_index([j for j, x in enumerate(l) if x == t])


def main():
    with open(data_list_file, "r") as f:
        data_list = json.load(f)

    for receptor in data_list["pdb_name"]:
        pcd = o3d.io.read_point_cloud(data_dir_pass + receptor + ".ply")
        pcd_denoised = pocket_denoising(pcd)
        o3d.io.write_point_cloud(save_dir + receptor + ".ply", pcd_denoised)


if __name__ == "__main__":
    main()
