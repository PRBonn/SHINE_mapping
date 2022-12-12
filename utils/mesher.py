import numpy as np
from tqdm import tqdm
import skimage.measure
import torch
import math
import open3d as o3d
import copy
import kaolin as kal
from utils.config import SHINEConfig
from utils.semantic_kitti_utils import *
from model.feature_octree import FeatureOctree
from model.decoder import Decoder

class Mesher():

    def __init__(self, config: SHINEConfig, octree: FeatureOctree, \
        geo_decoder: Decoder, sem_decoder: Decoder):

        self.config = config
    
        self.octree = octree
        self.geo_decoder = geo_decoder
        self.sem_decoder = sem_decoder
        self.device = config.device
        self.dtype = config.dtype
        self.world_scale = config.scale
    
    def query_points(self, coord, bs, query_sdf = True, query_sem = False, query_mask = True):
        """ query the sdf value, semantic label and marching cubes mask for points
        Args:
            coord: Nx3 torch tensor, the coordinates of all N (axbxc) query points in the scaled
                kaolin coordinate system [-1,1]
            bs: batch size for the inference
        Returns:
            sdf_pred: Ndim numpy array, signed distance value (scaled) at each query point
            sem_pred: Ndim numpy array, semantic label prediction at each query point
            mc_mask:  Ndim bool numpy array, marching cubes mask at each query point
        """
        # the coord torch tensor is already scaled in the [-1,1] coordinate system
        sample_count = coord.shape[0]
        iter_n = math.ceil(sample_count/bs)
        check_level = min(self.octree.featured_level_num, self.config.mc_vis_level)-1
        if query_sdf:
            sdf_pred = np.zeros(sample_count)
        else: 
            sdf_pred = None
        if query_sem:
            sem_pred = np.zeros(sample_count)
        else:
            sem_pred = None
        if query_mask:
            mc_mask = np.zeros(sample_count)
        else:
            mc_mask = None
        
        with torch.no_grad(): # eval step
            for n in tqdm(range(iter_n)):
                head = n*bs
                tail = min((n+1)*bs, sample_count)
                batch_coord = coord[head:tail]

                self.octree.get_indices_fast(batch_coord) 
                batch_feature = self.octree.query_feature(batch_coord)
                if query_sdf:
                    batch_sdf = -self.geo_decoder.sdf(batch_feature)
                    sdf_pred[head:tail] = batch_sdf.detach().cpu().numpy()
                if query_sem:
                    batch_sem = self.sem_decoder.sem_label(batch_feature)
                    sem_pred[head:tail] = batch_sem.detach().cpu().numpy()
                if query_mask:
                    # get the marching cubes mask
                    # hierarchical_indices: from bottom to top
                    check_level_indices = self.octree.hierarchical_indices[check_level] 
                    # if index is -1 for the level, then means the point is not valid under this level
                    mask_mc = check_level_indices >= 0
                    # all should be true (all the corner should be valid)
                    mask_mc = torch.all(mask_mc, dim=1)
                    mc_mask[head:tail] = mask_mc.detach().cpu().numpy()

                self.octree.set_zero()

        return sdf_pred, sem_pred, mc_mask

    def get_query_from_bbx(self, bbx, voxel_size):
        """ get grid query points inside a given bounding box (bbx)
        Args:
            bbx: open3d bounding box, in world coordinate system, with unit m 
            voxel_size: scalar, marching cubes voxel size with unit m
        Returns:
            coord: Nx3 torch tensor, the coordinates of all N (axbxc) query points in the scaled
                kaolin coordinate system [-1,1]
            voxel_num_xyz: 3dim numpy array, the number of voxels on each axis for the bbx
            voxel_origin: 3dim numpy array the coordinate of the bottom-left corner of the 3d grids 
                for marching cubes, in world coordinate system with unit m      
        """
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
    
    def assign_to_bbx(self, sdf_pred, sem_pred, mc_mask, voxel_num_xyz):
        """ assign the queried sdf, semantic label and marching cubes mask back to the 3D grids in the specified bounding box
        Args:
            sdf_pred: Ndim np.array
            sem_pred: Ndim np.array
            mc_mask:  Ndim bool np.array
            voxel_num_xyz: 3dim numpy array, the number of voxels on each axis for the bbx
        Returns:
            sdf_pred:  a*b*c np.array, 3d grids of sign distance values
            sem_pred:  a*b*c np.array, 3d grids of semantic labels
            mc_mask:   a*b*c np.array, 3d grids of marching cube masks, marching cubes only on where 
                the mask is true
        """
        if sdf_pred is not None:
            sdf_pred = sdf_pred.reshape(voxel_num_xyz[0], voxel_num_xyz[1], voxel_num_xyz[2])

        if sem_pred is not None:
            sem_pred = sem_pred.reshape(voxel_num_xyz[0], voxel_num_xyz[1], voxel_num_xyz[2])

        if mc_mask is not None:
            mc_mask = mc_mask.reshape(voxel_num_xyz[0], voxel_num_xyz[1], voxel_num_xyz[2]).astype(dtype=bool)

        return sdf_pred, sem_pred, mc_mask

    def mc_mesh(self, mc_sdf, mc_mask, voxel_size, mc_origin):
        """ use the marching cubes algorithm to get mesh vertices and faces
        Args:
            mc_sdf:  a*b*c np.array, 3d grids of sign distance values
            mc_mask: a*b*c np.array, 3d grids of marching cube masks, marching cubes only on where 
                the mask is true
            voxel_size: scalar, marching cubes voxel size with unit m
            mc_origin: 3*1 np.array, the coordinate of the bottom-left corner of the 3d grids for 
                marching cubes, in world coordinate system with unit m
        Returns:
            ([verts], [faces]), mesh vertices and triangle faces
        """
        # the input are all already numpy arraies
        verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
        try:       
            verts, faces, normals, values = skimage.measure.marching_cubes(
                mc_sdf, level=0.0, allow_degenerate=True, mask=mc_mask)
        except:
            pass

        verts = mc_origin + verts * voxel_size
        return verts, faces

    def recon_bbx_mesh(self, bbx, voxel_size, mesh_path, \
        estimate_sem = False, estimate_normal = True, filter_isolated_mesh = True):
        # reconstruct and save the (semantic) mesh from the feature octree the decoders within a
        # given bounding box.
        # bbx and voxel_size all with unit m, in world coordinate system

        coord, voxel_num_xyz, voxel_origin = self.get_query_from_bbx(bbx, voxel_size)
        sdf_pred, _, mc_mask = self.query_points(coord, self.config.infer_bs, True, False)
        mc_sdf, _, mc_mask = self.assign_to_bbx(sdf_pred, None, mc_mask, voxel_num_xyz)
        verts, faces = self.mc_mesh(mc_sdf, mc_mask, voxel_size, voxel_origin)

        # directly use open3d to get mesh
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts),
            o3d.utility.Vector3iVector(faces)
        )

        if estimate_sem:
            print("predict semantic labels of the vertices")
            verts_scaled = torch.tensor(verts * self.world_scale, dtype=self.dtype, device=self.device)
            _, verts_sem, _ = self.query_points(verts_scaled, self.config.infer_bs, False, True, False)
            verts_sem_list = list(verts_sem)
            verts_sem_rgb = [sem_kitti_color_map[sem_label] for sem_label in verts_sem_list]
            verts_sem_rgb = np.asarray(verts_sem_rgb)/255.0
            mesh.vertex_colors = o3d.utility.Vector3dVector(verts_sem_rgb)

        if estimate_normal:
            mesh.compute_vertex_normals()
        
        if filter_isolated_mesh:
            filter_cluster_min_tri = 1000
            # print("Cluster connected triangles")
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh.cluster_connected_triangles())
            triangle_clusters = np.asarray(triangle_clusters)
            cluster_n_triangles = np.asarray(cluster_n_triangles)
            cluster_area = np.asarray(cluster_area)

            # print("Remove the small clusters")
            mesh_0 = copy.deepcopy(mesh)
            triangles_to_remove = cluster_n_triangles[triangle_clusters] < filter_cluster_min_tri
            mesh_0.remove_triangles_by_mask(triangles_to_remove)
            mesh = mesh_0

        # write the mesh to ply file
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        print("save the mesh to %s\n" % (mesh_path))
