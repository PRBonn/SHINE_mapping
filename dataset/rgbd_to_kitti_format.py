import open3d as o3d
from tqdm import tqdm
import argparse
import re
import os
import numpy as np
import json
import shutil

def rgbd_to_kitti_format(args):

    ply_path = os.path.join(args.output_root, "rgbd_ply")
    os.makedirs(ply_path, exist_ok=True)

    # get pose
    pose_kitti_format_path = os.path.join(args.output_root, "poses.txt")
    if args.already_kitti_format_pose:
        shutil.copyfile(args.pose_file, pose_kitti_format_path) # don't directly copy, may have some issues
    else:
        poses_mat = load_poses(args.pose_file, with_head = False) # with_head = True for open3d provided Redwood dataset
        write_poses_kitti_format(poses_mat, pose_kitti_format_path)

    # get an example image 
    depth_img_files = sorted(os.listdir(args.depth_img_folder), key=alphanum_key)
    rgb_img_files = sorted(os.listdir(args.rgb_img_folder), key=alphanum_key)

    im_depth_example_path = os.path.join(args.depth_img_folder, depth_img_files[0])
    # print(im_depth_example_path)
    im_depth_example = o3d.io.read_image(im_depth_example_path)
    H, W = np.array(im_depth_example).shape[:2]
    print("Image size:", H, "x", W)
    
    # load the camera intrinsic parameters
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    depth_scale = 1000.
    if args.intrinsic_file == "":
        # use the default parameter
        # W=640, H=480, fx=fy=525.0, cx=319.5, cy=239.5
        print("Default intrinsic for PrimeSense used")
        intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
        # use this extrinsic matrix to rotate the image since frames captured with RealSense camera are upside down 
        extrinsic = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    else:
        if args.is_focal_file: # load the focal length only txt file # This is used for NeuralRGBD dataset
            focal = load_focal_length(args.intrinsic_file)
            print("Focal length:", focal)
            intrinsic.set_intrinsics(height=H,
                                     width=W,
                                     fx=focal,
                                     fy=focal,
                                     cx=(W-1.)/2.,
                                     cy=(H-1.)/2.)
            depth_scale = 1000.
            # use this extrinsic matrix to rotate the image since frames captured with RealSense camera are upside down 
            extrinsic = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        else:
            with open(args.intrinsic_file, 'r') as infile: # load intrinsic json file
                cam = json.load(infile)["camera"]
                intrinsic.set_intrinsics(height=cam["h"],
                                        width=cam["w"],
                                        fx=cam["fx"],
                                        fy=cam["fy"],
                                        cx=cam["cx"],
                                        cy=cam["cy"])
                depth_scale = cam["scale"]
                extrinsic = np.eye(4) # this is used for Replica dataset
    

    # get point cloud
    frame_count = 0
    for color_path, depth_path in tqdm(zip(rgb_img_files, depth_img_files)):
        color_path = os.path.join(args.rgb_img_folder, color_path)
        depth_path = os.path.join(args.depth_img_folder, depth_path)
        print(color_path)
        print(depth_path)
        
        im_color = o3d.io.read_image(color_path)
        im_depth = o3d.io.read_image(depth_path) 
        im_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(im_color, im_depth, depth_scale = depth_scale, depth_trunc = args.max_depth_m, convert_rgb_to_intensity=False) # not just gray
        im_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(im_rgbd, intrinsic, extrinsic)

        if args.vis_on:
            o3d.visualization.draw_geometries([im_pcd])
        frame_id_str = f'{frame_count:06d}'
        cur_filename = frame_id_str+".ply"
        cur_path = os.path.join(ply_path, cur_filename)
        o3d.io.write_point_cloud(cur_path, im_pcd)      

        frame_count+=1
    
    print("The rgbd dataset in KITTI format has been saved at %s", args.output_root)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [int(x) if x.isdigit() else x for x in re.split('([0-9]+)', s)]

def load_from_json(filename):
    """Load a dictionary from a JSON filename.
    Args:
        filename: The filename to load from.
    """
    assert filename.suffix == ".json"
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)

def load_focal_length(filepath):
    file = open(filepath, "r")
    return float(file.readline())

def load_poses(posefile, with_head = False):
    file = open(posefile, "r")
    lines = file.readlines()
    file.close()
    poses = []
    if not with_head:
        lines_per_matrix = 4
        skip_line = 0
    else: 
        lines_per_matrix = 5
        skip_line = 1
    for i in range(0, len(lines), lines_per_matrix):
        pose_floats = np.array([[float(x) for x in line.split()] for line in lines[i+skip_line:i+lines_per_matrix]])
        # print(pose_floats)
        poses.append(pose_floats)

    return poses

def write_poses_kitti_format(poses_mat, posefile):
    poses_vec = []
    for pose_mat in poses_mat:
        pose_vec= pose_mat.flatten()[0:12]
        poses_vec.append(pose_vec)
    np.savetxt(posefile, poses_vec, delimiter=' ')

def parser_json_sdf_studio_format(json_file):
    meta_data = load_from_json(json_file)
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--depth_img_folder', help="folder containing the depth images")
    parser.add_argument('--rgb_img_folder', help="folder containing the rgb images")
    parser.add_argument('--intrinsic_file', default="", help="path to the json file containing the camera intrinsic parameters")
    parser.add_argument('--pose_file', help="path to the txt file containing the camera pose at each frame")
    parser.add_argument('--output_root', help="path for outputing the kitti format data")
    parser.add_argument('--max_depth_m', type=float, default=5.0, help="maximum depth to be used")
    parser.add_argument('--is_focal_file', type=str2bool, nargs='?', default=True, \
        help="is the input intrinsic file a txt file containing only the focus length (as the Neural RGBD data format)\
              or the json file containing all the intrinsic parameters (as the Replica format)")
    parser.add_argument('--already_kitti_format_pose', type=str2bool, nargs='?', default=False, \
        help="is the input pose file already in KITTI pose format (also as the Replica format)\
              or the input pose file is in a 4dim transformation form (as the Neural RGBD data format)")
    parser.add_argument('--vis_on', type=str2bool, nargs='?', default=False)
    args = parser.parse_args()
    
    rgbd_to_kitti_format(args)