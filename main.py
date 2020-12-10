import open3d as o3d
import docking 
import scoring 
import func
import json
import numpy as np
import time


# dataset pass
data_dir_pass = "../../data/dud38_ply/"
# read dud38_list
with open("dud38_list.json", "r") as f:
    data_list = json.load(f)
# read decoy_list
with open("decoy_list.json", "r") as f:
    decoy_list = json.load(f)


def main(num):
    """num: is target(protein pocket) number 
    """
    target = func.init_pcd(data_dir_pass + data_list["pdb_name"][num] + ".ply")

    score = []

    # screening
    # data in "dud38"
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

    # screening
    # data in "decoy"
    """
    for ligands in decoy_list["decoy"]:
        for ligand in ligands["ligand"]:
            # load ligand data
            source = func.init_pcd(data_dir_pass + "decoy/" + ligand + ".ply")
            # docking
            docking.docking(target, source)
            # scoring
            score.append(scoring.scoring(source, target))

    return np.argsort(np.array(score))
    """


def virtual_screening():

    target_num = [i for i in range(38)]

    top10 = 0
    top5 = 0
    total = 0
    start_time = time.time()
    for num in target_num:
        checker = main(num)
        if num in checker[:10]:
            top10 = top10 + 1
        if num in checker[:5]:
            top5 = top5 + 1
        total = total + 1
        print(num, ":", checker[:10])
        print("top10:", top10/total, "top5:", top5/total)
        print("time:", time.time()-start_time)

def test():
    target_num = 23

    checker = main(target_num)
    print(target_num, ":", checker[:10])
    if target_num in checker[:10]:
        print("top10: finded")
    

if __name__ == "__main__":
    test()

