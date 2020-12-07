import open3d as o3d
import docking 
import scoring 
import func
import json
import numpy as np


data_list_file = "dud38_list.json"
data_dir_pass = "../../data/dud38_ply/"


def main(num):
    with open(data_list_file, "r") as f:
        data_list = json.load(f)

    target = func.init_pcd(data_dir_pass + data_list["pdb_name"][num] + ".ply")

    score = []

    # screening
    i = 0
    for ligand in data_list["ligand_name"]:
        source = func.init_pcd(data_dir_pass + ligand + ".ply")

        # docking
        docking.docking(target, source)

        # scoring
        score.append(scoring.scoring(source, target))
        # score.append(docking.docking(target, source).inlier_rmse)

        """
        # save pose 
        address = "result/target"+str(num)+"/withligand"+str(i)+".ply"
        two_pcd = source.pcd_full_points + target.pcd_full_points
        o3d.io.write_point_cloud(address, two_pcd)
        i = i + 1
        """

    return np.argsort(np.array(score))


def virtual_screening():

    target_num = [i for i in range(38)]

    top10 = 0
    top5 = 0
    total = 0
    for num in target_num:
        checker = main(num)
        if num in checker[:10]:
            top10 = top10 + 1
        if num in checker[:5]:
            top5 = top5 + 1
        total = total + 1
        print(num, ":", checker[:10])
        print("top10:", top10/total, "top5:", top5/total)

def test():
    target_num = 3

    checker = main(target_num)
    print(target_num, ":", checker[:10])
    

if __name__ == "__main__":
    virtual_screening()

