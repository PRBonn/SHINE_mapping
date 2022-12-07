import numpy as np
from tqdm import tqdm
import skimage.measure
import torch
import math
import open3d as o3d
import copy
import kaolin as kal
from utils.config import SHINEConfig
from model.feature_octree import FeatureOctree
from model.decoder import Decoder

class Mesher():

    def __init__(self, config: SHINEConfig, octree: FeatureOctree, decoder: Decoder):

        self.config = config
    
        self.octree = octree
        self.decoder = decoder
        self.device = config.device
        self.dtype = config.dtype
        self.world_scale = config.scale
    
    def query_sdf(self, coord, bs):
        # the coord torch tensor is already scaled in the [-1,1] coordinate system
        sample_count = coord.shape[0]
        iter_n = math.ceil(sample_count/bs)
        check_level = min(self.octree.featured_level_num, self.config.mc_vis_level)-1
        sdf_pred = np.zeros(sample_count)
        mc_mask = np.zeros(sample_count)
        
        with torch.no_grad(): # eval step
            for n in tqdm(range(iter_n)):
                head = n*bs
                tail = min((n+1)*bs, sample_count)
                batch_coord = coord[head:tail]

                self.octree.get_indices_fast(batch_coord) 
                batch_feature = self.octree.query_feature(batch_coord)
                batch_sdf = -self.decoder(batch_feature)
                self.octree.set_zero()
                
                # get the marching cubes mask
                # hierarchical_indices: from bottom to top
                check_level_indices = self.octree.hierarchical_indices[check_level] 
                # if index is -1 for the level, then means the point is not valid under this level
                mask_mc = check_level_indices >= 0
                # all should be true (all the corner should be valid)
                mask_mc = torch.all(mask_mc, dim=1)

                sdf_pred[head:tail] = batch_sdf.detach().cpu().numpy()
                mc_mask[head:tail] = mask_mc.detach().cpu().numpy()

        return sdf_pred, mc_mask

    def get_query_from_bbx(self, bbx, voxel_size):
        # bbx and voxel_size are all in the world coordinate system
        min_bound = bbx.get_min_bound()
        max_bound = bbx.get_max_bound()
        len_xyz = max_bound - min_bound
        voxel_num_xyz = (np.ceil(len_xyz/voxel_size)+self.config.pad_voxel*2).astype(np.int_)
        voxel_origin = min_bound-self.config.pad_voxel*voxel_size

        x = torch.arange(voxel_num_xyz[0], dtype=torch.int16, device=self.device)
        y = torch.arange(voxel_num_xyz[1], dtype=torch.int16, device=self.device)
        z = torch.arange(voxel_num_xyz[2], dtype=torch.int16, device=self.device)

        # order: [0,0,0], [0,0,1], [0,0,2], [0,1,0], [0,1,1], [0,1,2] ...
        x, y, z = torch.meshgrid(x, y, z, indexing='ij') 
        # get the vector of all the grid point's 3D coordinates
        coord = torch.stack((x.flatten(), y.flatten(), z.flatten())).transpose(0, 1).float()
        # transform to world coordinate system
        coord *= voxel_size
        coord += torch.tensor(voxel_origin, dtype=self.dtype, device=self.device)
        # scaling to the [-1, 1] coordinate system
        coord *= self.world_scale
        
        return coord, voxel_num_xyz, voxel_origin
    
    def assign_sdf_to_bbx(self, sdf_pred, mc_mask, voxel_num_xyz):
        mc_sdf = sdf_pred.reshape(voxel_num_xyz[0], voxel_num_xyz[1], voxel_num_xyz[2])
        mc_mask = mc_mask.reshape(voxel_num_xyz[0], voxel_num_xyz[1], voxel_num_xyz[2]).astype(dtype=bool)
        
        return mc_sdf, mc_mask

    
    def mc_mesh(self, mc_sdf, mc_mask, voxel_size, mc_origin):
        # the input are all already numpy arraies
        verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
        try:       
            verts, faces, normals, values = skimage.measure.marching_cubes(
                mc_sdf, level=0.0, allow_degenerate=True, mask=mc_mask)
        except:
            pass

        verts = mc_origin + verts * voxel_size

        # directly use open3d
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts),
            o3d.utility.Vector3iVector(faces)
        )

        return mesh

    def recon_bbx_mesh(self, bbx, voxel_size, mesh_path, \
        estimate_normal = True, filter_isolated_mesh = True):
        coord, voxel_num_xyz, voxel_origin = self.get_query_from_bbx(bbx, voxel_size)
        sdf_pred, mc_mask = self.query_sdf(coord, self.config.infer_bs)
        mc_sdf, mc_mask = self.assign_sdf_to_bbx(sdf_pred, mc_mask, voxel_num_xyz)
        mesh = self.mc_mesh(mc_sdf, mc_mask, voxel_size, voxel_origin)

        if estimate_normal:
            mesh.compute_vertex_normals()
        
        if filter_isolated_mesh:
            filter_cluster_min_tri = 1000
            # print("Cluster connected triangles")
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh.cluster_connected_triangles())
            triangle_clusters = np.asarray(triangle_clusters)
            cluster_n_triangles = np.asarray(cluster_n_triangles)
            cluster_area = np.asarray(cluster_area)

            print("Remove the small clusters")
            mesh_0 = copy.deepcopy(mesh)
            triangles_to_remove = cluster_n_triangles[triangle_clusters] < filter_cluster_min_tri
            mesh_0.remove_triangles_by_mask(triangles_to_remove)
            mesh = mesh_0

        # write the mesh to ply file
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        print("save the mesh to %s\n" % (mesh_path))
