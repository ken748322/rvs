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
    for ligand in data_list["ligand_name"]:
        source = func.init_pcd(data_dir_pass + ligand + ".ply")

        # docking
        docking.docking(target, source)

        # scoring
        score.append(scoring.scoring(source, target))

    return np.argsort(np.array(score))


if __name__ == "__main__":
    print(main(13))
