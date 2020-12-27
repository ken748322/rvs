import open3d as o3d  
import numpy as np 
from sklearn import neighbors
from shapely.geometry import Polygon
from shapely.geometry import Point

def which_points_inside_source(source, target):

    # point-cloudsをnp.arrayに変更
    s_points = np.asarray(source.pcd.points)
    t_points = np.asarray(target.pcd.points)

    # リガンドを0.5ずつ分割する
    cut_surface = [i for i in np.arange(s_points[:,2].max(), s_points[:,2].min(), -0.2)]

    # 内外判定
    points_inside_idx = []

    for l in range(len(cut_surface)-1):
    # c = 各層に対応するindex
        s_c = list((s_points[:,2] > cut_surface[l+1]) & (s_points[:,2] < cut_surface[l]))
        s_c = [i for i, x in enumerate(s_c) if x == True]

        t_c = list((t_points[:,2] > cut_surface[l+1]) & (t_points[:,2] < cut_surface[l]))
        t_c = [i for i, x in enumerate(t_c) if x == True]

        # 抽出
        layer = s_points[s_c][:, 0:2]
        points = t_points[t_c][:, 0:2]

        # ポリゴンを作成する
        # 近傍探索
        if len(layer)>2:
            tree = neighbors.KDTree(layer)
            order = [] 
            q = 0  # query
            k = 2  # 最近傍の個数

            # 最近傍探索
            while len(order) != len(layer):
                _, idx = tree.query([layer[q]], k=k)
                if idx[0,k-1] not in order:
                    order.append(idx[0,k-1])
                    q = idx[0,k-1]
                    k = 2
                else:
                    k = k + 1

            # ポリゴン作成
            polygon = Polygon(layer[order])

            # ポリゴン内に存在する点のindexを返す
            in_or_out = [Point(p).within(polygon) for p in points]
            c = [i for i, x in enumerate(in_or_out) if x == True]
            for i in c:
                points_inside_idx.append(t_c[i])
    
    return points_inside_idx, t_points[points_inside_idx]


def scoring(source, target):
    """compute the squared error with the nearest neighbor in the target of the source
    """
    # down sample し直す
    source.reset_down_sample()
    target.reset_down_sample()
    voxel_size = 0.2
    source.down_sample(voxel_size)
    target.down_sample(voxel_size)

    # source の最近傍点の二乗誤差の平均
    error = np.average(source.pcd.compute_point_cloud_distance(target.pcd))

    # souce 内にあるtargetの点を抽出
    idx, points_inside = which_points_inside_source(source, target)

    # tmp = source内部にあるtargetの点
    tmp_pcd = o3d.geometry.PointCloud()
    tmp_pcd.points = o3d.utility.Vector3dVector(points_inside)

    # source内にあるtargetに重みをつける
    inside_error = np.asarray(tmp_pcd.compute_point_cloud_distance(source.pcd))
    error = error + np.average(inside_error[list(inside_error > 0.2)]) * 10

    return error
