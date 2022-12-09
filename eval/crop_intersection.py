from eval_utils import crop_intersection

# This file presents an example to crop the ground truth point cloud to the intersection part of all
# the compared method's mesh reconstruction

gt_pcd_path = "xxx/mai_city/01/gt_map_pc_mai.ply"

pred_vdb_path = "xxx/mai_city/01/baseline/vdb_fusion/mesh_vdb_10cm.ply"

pred_puma_path = "xxx/mai_city/01/baseline/puma/mesh_puma_l10.ply"

pred_voxblox_path = "xxx/mai_city/01/baseline/voxblox/mesh_voxblox_10cm.ply"

pred_shine_path = "xxx/mai_city/01/mesh_shine_10cm.ply"

preds_path = [pred_vdb_path, pred_puma_path, pred_voxblox_path, pred_shine_path]

crop_gt_pcd_path = "xxx/mai_city/01/gt_map_pc_mai_crop_intersection.ply"

crop_intersection(gt_pcd_path, preds_path, crop_gt_pcd_path, dist_thre=0.2)