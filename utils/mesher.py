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

    # TODO: add methods to reconstruct large scale meshs without memory issues [marching cubes in blocks]

    def __init__(self, config: SHINEConfig, octree: FeatureOctree, \
        geo_decoder: Decoder, sem_decoder: Decoder):

        self.config = config
    
        self.octree = octree
        self.geo_decoder = geo_decoder
        self.sem_decoder = sem_decoder
        self.device = config.device
        self.cur_device = self.device
        self.dtype = config.dtype
        self.world_scale = config.scale

        self.ts = 0 # query timestamp when conditioned on time

        self.global_transform = np.eye(4)
    
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
            if iter_n > 1:
                for n in tqdm(range(iter_n)):
                    head = n*bs
                    tail = min((n+1)*bs, sample_count)
                    batch_coord = coord[head:tail, :]
                    if self.cur_device == "cpu" and self.device == "cuda":
                        batch_coord = batch_coord.cuda()
                    batch_feature = self.octree.query_feature(batch_coord, True) # query features
                    if query_sdf:
                        if not self.config.time_conditioned:
                            batch_sdf = -self.geo_decoder.sdf(batch_feature)
                        else:
                            batch_sdf = -self.geo_decoder.time_conditionded_sdf(batch_feature, self.ts * torch.ones(batch_feature.shape[0], 1).cuda())
                        sdf_pred[head:tail] = batch_sdf.detach().cpu().numpy()
                    if query_sem:
                        batch_sem = self.sem_decoder.sem_label(batch_feature)
                        sem_pred[head:tail] = batch_sem.detach().cpu().numpy()
                    if query_mask:
                        # get the marching cubes mask
                        # hierarchical_indices: bottom-up
                        check_level_indices = self.octree.hierarchical_indices[check_level] 
                        # print(check_level_indices)
                        # if index is -1 for the level, then means the point is not valid under this level
                        mask_mc = check_level_indices >= 0
                        # print(mask_mc.shape)
                        # all should be true (all the corner should be valid)
                        mask_mc = torch.all(mask_mc, dim=1)
                        mc_mask[head:tail] = mask_mc.detach().cpu().numpy()
                        # but for scimage's marching cubes, the top right corner's mask should also be true to conduct marching cubes
            else:
                feature = self.octree.query_feature(coord, True)
                if query_sdf:
                    if not self.config.time_conditioned:
                        sdf_pred = -self.geo_decoder.sdf(feature).detach().cpu().numpy()
                    else: # just for a quick test
                        sdf_pred = -self.geo_decoder.time_conditionded_sdf(feature, self.ts * torch.ones(feature.shape[0], 1).cuda()).detach().cpu().numpy()
                if query_sem:
                    sem_pred = self.sem_decoder.sem_label(feature).detach().cpu().numpy()
                if query_mask:
                    # get the marching cubes mask
                    check_level_indices = self.octree.hierarchical_indices[check_level] 
                    # if index is -1 for the level, then means the point is not valid under this level
                    mask_mc = check_level_indices >= 0
                    # all should be true (all the corner should be valid)
                    mc_mask = torch.all(mask_mc, dim=1).detach().cpu().numpy()

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
        # pad an additional voxel underground to gurantee the reconstruction of ground
        voxel_origin[2]-=voxel_size
        voxel_num_xyz[2]+=1

        voxel_count_total = voxel_num_xyz[0] * voxel_num_xyz[1] * voxel_num_xyz[2]
        if voxel_count_total > 1e8: # TODO: avoid gpu memory issue, dirty fix
            self.cur_device = "cpu" # firstly save in cpu memory (which would be larger than gpu's)
            print("too much query points, use cpu memory")
        x = torch.arange(voxel_num_xyz[0], dtype=torch.int16, device=self.cur_device)
        y = torch.arange(voxel_num_xyz[1], dtype=torch.int16, device=self.cur_device)
        z = torch.arange(voxel_num_xyz[2], dtype=torch.int16, device=self.cur_device)

        # order: [0,0,0], [0,0,1], [0,0,2], [0,1,0], [0,1,1], [0,1,2] ...
        x, y, z = torch.meshgrid(x, y, z, indexing='ij') 
        # get the vector of all the grid point's 3D coordinates
        coord = torch.stack((x.flatten(), y.flatten(), z.flatten())).transpose(0, 1).float()
        # transform to world coordinate system
        coord *= voxel_size
        coord += torch.tensor(voxel_origin, dtype=self.dtype, device=self.cur_device)
        # scaling to the [-1, 1] coordinate system
        coord *= self.world_scale
        
        return coord, voxel_num_xyz, voxel_origin
    
    def generate_sdf_map(self, coord, sdf_pred, mc_mask, map_path):
        device = o3d.core.Device("CPU:0")
        dtype = o3d.core.float32
        sdf_map_pc = o3d.t.geometry.PointCloud(device)

        # scaling back to the world coordinate system
        coord /= self.world_scale
        coord_np = coord.detach().cpu().numpy()

        sdf_pred_world = sdf_pred * self.config.logistic_gaussian_ratio*self.config.sigma_sigmoid_m # convert to unit: m

        # the sdf (unit: m) would be saved in the intensity channel
        sdf_map_pc.point['positions'] = o3d.core.Tensor(coord_np, dtype, device)
        sdf_map_pc.point['intensities'] = o3d.core.Tensor(np.expand_dims(sdf_pred_world, axis=1), dtype, device) # scaled sdf prediction
        if mc_mask is not None:
            # the marching cubes mask would be saved in the labels channel (indicating the hierarchical position in the octree)
            sdf_map_pc.point['labels'] = o3d.core.Tensor(np.expand_dims(mc_mask, axis=1), o3d.core.int32, device) # mask

        # global transform (to world coordinate system) before output
        sdf_map_pc.transform(self.global_transform)
        o3d.t.io.write_point_cloud(map_path, sdf_map_pc, print_progress=False)
        print("save the sdf map to %s" % (map_path))
    
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
            # mc_mask[:,:,0:1] = True # TODO: dirty fix for the ground issue 

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
        print("Marching cubes ...")
        # the input are all already numpy arraies
        verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
        try:       
            verts, faces, normals, values = skimage.measure.marching_cubes(
                mc_sdf, level=0.0, allow_degenerate=False, mask=mc_mask)
        except:
            pass

        verts = mc_origin + verts * voxel_size
        return verts, faces

    def estimate_vertices_sem(self, mesh, verts, filter_free_space_vertices = True):
        print("predict semantic labels of the vertices")
        verts_scaled = torch.tensor(verts * self.world_scale, dtype=self.dtype, device=self.device)
        _, verts_sem, _ = self.query_points(verts_scaled, self.config.infer_bs, False, True, False)
        verts_sem_list = list(verts_sem)
        verts_sem_rgb = [sem_kitti_color_map[sem_label] for sem_label in verts_sem_list]
        verts_sem_rgb = np.asarray(verts_sem_rgb)/255.0
        mesh.vertex_colors = o3d.utility.Vector3dVector(verts_sem_rgb)

        # filter the freespace vertices
        if filter_free_space_vertices:
            non_freespace_idx = verts_sem <= 0
            mesh.remove_vertices_by_mask(non_freespace_idx)
        
        return mesh

    def filter_isolated_vertices(self, mesh, filter_cluster_min_tri = 300):
        # print("Cluster connected triangles")
        triangle_clusters, cluster_n_triangles, _ = (mesh.cluster_connected_triangles())
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        # cluster_area = np.asarray(cluster_area)
        # print("Remove the small clusters")
        # mesh_0 = copy.deepcopy(mesh)
        triangles_to_remove = cluster_n_triangles[triangle_clusters] < filter_cluster_min_tri
        mesh.remove_triangles_by_mask(triangles_to_remove)
        # mesh = mesh_0
        return mesh

    def recon_bbx_mesh(self, bbx, voxel_size, mesh_path, map_path, \
        save_map = False, estimate_sem = False, estimate_normal = True, \
        filter_isolated_mesh = True, filter_free_space_vertices = True):
        
        # reconstruct and save the (semantic) mesh from the feature octree the decoders within a
        # given bounding box.
        # bbx and voxel_size all with unit m, in world coordinate system

        coord, voxel_num_xyz, voxel_origin = self.get_query_from_bbx(bbx, voxel_size)
        sdf_pred, _, mc_mask = self.query_points(coord, self.config.infer_bs, True, False, self.config.mc_mask_on)
        if save_map:
            self.generate_sdf_map(coord, sdf_pred, mc_mask, map_path)
        mc_sdf, _, mc_mask = self.assign_to_bbx(sdf_pred, None, mc_mask, voxel_num_xyz)
        verts, faces = self.mc_mesh(mc_sdf, mc_mask, voxel_size, voxel_origin)

        # directly use open3d to get mesh
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts),
            o3d.utility.Vector3iVector(faces)
        )

        if estimate_sem: 
            mesh = self.estimate_vertices_sem(mesh, verts, filter_free_space_vertices)

        if estimate_normal:
            mesh.compute_vertex_normals()
        
        if filter_isolated_mesh:
            mesh = self.filter_isolated_vertices(mesh, self.config.min_cluster_vertices)

        # global transform (to world coordinate system) before output
        mesh.transform(self.global_transform)

        # write the mesh to ply file
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        print("save the mesh to %s\n" % (mesh_path))

        return mesh

    # reconstruct the map sparsely using the octree, only query the sdf at certain level ($query_level) of the octree
    # much faster and also memory-wise more efficient
    def recon_octree_mesh(self, query_level, mc_res_m, mesh_path, map_path, \
                          save_map = False, estimate_sem = False, estimate_normal = True, \
                          filter_isolated_mesh = True, filter_free_space_vertices = True): 

        nodes_coord_scaled = self.octree.get_octree_nodes(query_level) # query level top-down
        nodes_count = nodes_coord_scaled.shape[0]
        min_nodes = np.min(nodes_coord_scaled, 0)
        max_nodes = np.max(nodes_coord_scaled, 0)

        node_res_scaled = 2**(1-query_level) # voxel size for queried octree node in [-1,1] coordinate system
        # marching cube's voxel size should be evenly divisible by the queried octree node's size
        voxel_count_per_side_node = np.ceil(node_res_scaled / self.world_scale / mc_res_m).astype(dtype=int) 
        # assign coordinates for the queried octree node
        x = torch.arange(voxel_count_per_side_node, dtype=torch.int16, device=self.device)
        y = torch.arange(voxel_count_per_side_node, dtype=torch.int16, device=self.device)
        z = torch.arange(voxel_count_per_side_node, dtype=torch.int16, device=self.device)
        node_box_size = (np.ones(3)*voxel_count_per_side_node).astype(dtype=int)

        # order: [0,0,0], [0,0,1], [0,0,2], [0,1,0], [0,1,1], [0,1,2] ...
        x, y, z = torch.meshgrid(x, y, z, indexing='ij') 
        # get the vector of all the grid point's 3D coordinates
        coord = torch.stack((x.flatten(), y.flatten(), z.flatten())).transpose(0, 1).float() 
        mc_res_scaled = node_res_scaled / voxel_count_per_side_node # voxel size for marching cubes in [-1,1] coordinate system
        # transform to [-1,1] coordinate system
        coord *= mc_res_scaled

        # the voxel count for the whole map
        voxel_count_per_side = ((max_nodes - min_nodes)/mc_res_scaled+voxel_count_per_side_node).astype(int)
        # initialize the whole map
        query_grid_sdf = np.zeros((voxel_count_per_side[0], voxel_count_per_side[1], voxel_count_per_side[2]), dtype=np.float16) # use float16 to save memory
        query_grid_mask = np.zeros((voxel_count_per_side[0], voxel_count_per_side[1], voxel_count_per_side[2]), dtype=bool)  # mask off

        for node_idx in tqdm(range(nodes_count)):
            node_coord_scaled = nodes_coord_scaled[node_idx, :]
            cur_origin = torch.tensor(node_coord_scaled - 0.5 * (node_res_scaled - mc_res_scaled), device=self.device)
            cur_coord = coord.clone()
            cur_coord += cur_origin
            cur_sdf_pred, _, cur_mc_mask = self.query_points(cur_coord, self.config.infer_bs, True, False, self.config.mc_mask_on)
            cur_sdf_pred, _, cur_mc_mask = self.assign_to_bbx(cur_sdf_pred, None, cur_mc_mask, node_box_size)
            shift_coord = (node_coord_scaled - min_nodes)/node_res_scaled
            shift_coord = (shift_coord*voxel_count_per_side_node).astype(int)
            query_grid_sdf[shift_coord[0]:shift_coord[0]+voxel_count_per_side_node, shift_coord[1]:shift_coord[1]+voxel_count_per_side_node, shift_coord[2]:shift_coord[2]+voxel_count_per_side_node] = cur_sdf_pred
            query_grid_mask[shift_coord[0]:shift_coord[0]+voxel_count_per_side_node, shift_coord[1]:shift_coord[1]+voxel_count_per_side_node, shift_coord[2]:shift_coord[2]+voxel_count_per_side_node] = cur_mc_mask

        mc_voxel_size = mc_res_scaled / self.world_scale
        mc_voxel_origin = (min_nodes - 0.5 * (node_res_scaled - mc_res_scaled)) / self.world_scale

        # if save_map: # ignore it now, too much for the memory
        #     # query_grid_coord 
        #     self.generate_sdf_map(query_grid_coord, query_grid_sdf, query_grid_mask, map_path)

        verts, faces = self.mc_mesh(query_grid_sdf, query_grid_mask, mc_voxel_size, mc_voxel_origin)
        # directly use open3d to get mesh
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts),
            o3d.utility.Vector3iVector(faces)
        )

        if estimate_sem: 
            mesh = self.estimate_vertices_sem(mesh, verts, filter_free_space_vertices)

        if estimate_normal:
            mesh.compute_vertex_normals()
        
        if filter_isolated_mesh:
            mesh = self.filter_isolated_vertices(mesh)

        # global transform (to world coordinate system) before output
        mesh.transform(self.global_transform)

        # write the mesh to ply file
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        print("save the mesh to %s\n" % (mesh_path))

        return mesh