import open3d as o3d
import os

path = "/root/studio/project/puma_ma/puma/apps/data/ncd/quad/pcd/" #文件夹目录
path_2 = "/root/studio/project/puma_ma/puma/apps/data/ncd/quad/" 
files = os.listdir(path) #得到文件夹下的所有文件名称

for file_name in files:
    pcd = o3d.io.read_point_cloud(path + file_name)

    out_put_path = path_2 + "ply/" + str(file_name[:-3]) + "ply"
    
    o3d.io.write_point_cloud(out_put_path, pcd)