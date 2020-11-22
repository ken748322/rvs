import open3d as o3d
from docking import *
from scoring import * 
import func
import json


data_list_file = "dud38_list.json"
data_dir_pass = "../../data/dud38_ply/"

with open(data_list_file, "r") as f:
    data_list = json.load(f)

target = func.init_pcd(data_dir_pass + data_list["pdb_name"][35] + ".ply")

score = []

# screening
for ligand in data_list["ligand_name"]:
    source = func.init_pcd(data_dir_pass + ligand + ".ply")

    # docking
    docking(target, source)

    # scoring
    score.append(scoring(source, target))

print(np.argsort(np.array(score)))
