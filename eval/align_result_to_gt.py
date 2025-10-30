# --------------------------------------------------------
# Evaluation utilities. The code is adapted from SLAM3R: 
# https://github.com/PKU-VCL-3DV/SLAM3R/blob/main/evaluation/eval_recon.py
# --------------------------------------------------------
import os
from os.path import join 
import json
import trimesh
import argparse
import numpy as np
import random
import matplotlib.pyplot as pl
pl.ion()
import open3d as o3d
from scipy.spatial import cKDTree as KDTree
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Inference on a pair of images from ScanNet++")
parser.add_argument("--save_vis", action="store_true")
parser.add_argument("--seed", type=int, default=42, help="seed for python random")
parser.add_argument("--icp", type=str, default='plain', help='')
parser.add_argument("--gt_pcd", type=str, required=True, help='')
parser.add_argument("--pred_pcd", type=str, required=True, help='')
parser.add_argument("--n_samples", type=int, default=100, help='number of points per RANSAC hypothesis')
parser.add_argument("--n_hypotheses", type=int, default=512, help='number of RANSAC hypotheses')
parser.add_argument("--inlier_threshold", type=float, default=0.1, help='RANSAC inlier distance threshold')
parser.add_argument("--use_fpfh", action="store_true", help='Use FPFH features for initial alignment')
parser.add_argument("--skip_icp", action="store_true", help='Skip ICP refinement')

def load_and_sample_gt_mesh(glb_path, num_points=100000):
    # Load GLB mesh
    mesh = o3d.io.read_triangle_mesh(glb_path)
    assert mesh.has_triangles(), "GT .glb file does not contain a triangle mesh!"

    # Uniformly sample points on surface
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)

    return np.asarray(pcd.points, dtype=np.float32)

def load_pred_pcd_from_glb(glb_path):
    scene = trimesh.load(glb_path, force='scene')

    # Extract all geometries' vertices (there should be only one)
    vertices = []
    for name, geom in scene.geometry.items():
        if hasattr(geom, "vertices"):
            vertices.append(np.asarray(geom.vertices, dtype=np.float32))
    
    if len(vertices) == 0:
        raise RuntimeError(f"No vertex geometry found in {glb_path}")
    if len(vertices) > 1:
        print(f"[WARN] Multiple geometries in pred glb, concatenating {len(vertices)} vertex sets")

    pts = np.vstack(vertices)
    return pts

def analyze_point_clouds(src, tgt):
    """Analyze point cloud properties to diagnose alignment issues"""
    print("\n" + "="*60)
    print("POINT CLOUD ANALYSIS")
    print("="*60)
    
    # Basic statistics
    print("\nSource Point Cloud:")
    print(f"  Points: {len(src)}")
    print(f"  Min: [{src.min(axis=0)}]")
    print(f"  Max: [{src.max(axis=0)}]")
    print(f"  Mean: [{src.mean(axis=0)}]")
    print(f"  Std: [{src.std(axis=0)}]")
    src_extent = src.max(axis=0) - src.min(axis=0)
    print(f"  Extent: [{src_extent}]")
    print(f"  Diagonal: {np.linalg.norm(src_extent):.4f}")
    
    print("\nTarget Point Cloud:")
    print(f"  Points: {len(tgt)}")
    print(f"  Min: [{tgt.min(axis=0)}]")
    print(f"  Max: [{tgt.max(axis=0)}]")
    print(f"  Mean: [{tgt.mean(axis=0)}]")
    print(f"  Std: [{tgt.std(axis=0)}]")
    tgt_extent = tgt.max(axis=0) - tgt.min(axis=0)
    print(f"  Extent: [{tgt_extent}]")
    print(f"  Diagonal: {np.linalg.norm(tgt_extent):.4f}")
    
    # Scale ratio
    src_scale = np.linalg.norm(src_extent)
    tgt_scale = np.linalg.norm(tgt_extent)
    scale_ratio = tgt_scale / src_scale
    print(f"\nScale ratio (target/source): {scale_ratio:.4f}")
    
    # Center distance
    center_dist = np.linalg.norm(src.mean(axis=0) - tgt.mean(axis=0))
    print(f"Distance between centers: {center_dist:.4f}")
    
    print("="*60 + "\n")
    
    return {
        'src_scale': src_scale,
        'tgt_scale': tgt_scale,
        'scale_ratio': scale_ratio,
        'center_dist': center_dist
    }

def normalize_point_cloud(pcd):
    """Normalize point cloud to unit sphere centered at origin"""
    center = pcd.mean(axis=0)
    pcd_centered = pcd - center
    scale = np.max(np.linalg.norm(pcd_centered, axis=1))
    pcd_normalized = pcd_centered / scale
    return pcd_normalized, center, scale

def denormalize_transformation(T, src_center, src_scale, tgt_center, tgt_scale):
    """Convert transformation from normalized space back to original space"""
    # T transforms normalized source to normalized target
    # We need: original_source -> normalized -> transformed -> denormalized -> original_target
    
    # Build complete transformation
    T_full = np.eye(4)
    
    # 1. Center and scale source
    T_normalize_src = np.eye(4)
    T_normalize_src[:3, 3] = -src_center
    T_scale_src = np.eye(4)
    T_scale_src[:3, :3] = np.eye(3) / src_scale
    
    # 2. Apply learned transformation
    # 3. Denormalize to target
    T_scale_tgt = np.eye(4)
    T_scale_tgt[:3, :3] = np.eye(3) * tgt_scale
    T_denormalize_tgt = np.eye(4)
    T_denormalize_tgt[:3, 3] = tgt_center
    
    # Compose all transformations
    T_full = T_denormalize_tgt @ T_scale_tgt @ T @ T_scale_src @ T_normalize_src
    
    return T_full

def fpfh_registration(source, target, voxel_size=0.05):
    """Use FPFH features for initial alignment"""
    print("Computing FPFH features for global registration...")
    
    # Convert to Open3D
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source)
    
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target)
    
    # Downsample
    source_down = source_pcd.voxel_down_sample(voxel_size)
    target_down = target_pcd.voxel_down_sample(voxel_size)
    
    # Estimate normals
    radius_normal = voxel_size * 2
    source_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    target_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    # Compute FPFH features
    radius_feature = voxel_size * 5
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    
    # RANSAC registration
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    
    print(f"FPFH registration: fitness={result.fitness:.4f}, inlier_rmse={result.inlier_rmse:.4f}")
    
    return result.transformation

def umeyama_alignment(X, Y, use_median=False):
    """
    Perform Umeyama alignment to align two point sets with potential size differences.

    Parameters:
    X (numpy.ndarray): Source point set with shape (N, D).
    Y (numpy.ndarray): Target point set with shape (N, D).
    use_median (bool): Use median instead of mean for centroid (more robust to outliers)

    Returns:
    T (numpy.ndarray): Transformation matrix (D+1, D+1) that aligns X to Y.
    """
    
    # Calculate centroids - use mean for better accuracy with larger samples
    if use_median and X.shape[0] < 10:
        centroid_X = np.median(X, axis=0)
        centroid_Y = np.median(Y, axis=0)
    else:
        centroid_X = np.mean(X, axis=0)
        centroid_Y = np.mean(Y, axis=0)

    # Center the point sets
    X_centered = X - centroid_X
    Y_centered = Y - centroid_Y

    # Solve rotation using SVD with rectification
    S = np.dot(X_centered.T, Y_centered)
    U, _, VT = np.linalg.svd(S)
    rectification = np.eye(3)
    rectification[-1,-1] = np.linalg.det(VT.T @ U.T)
    R = VT.T @ rectification @ U.T 

    # Scale factor - use mean for larger samples
    if use_median and X.shape[0] < 10:
        sx = np.median(np.linalg.norm(X_centered, axis=1))
        sy = np.median(np.linalg.norm(Y_centered, axis=1))
    else:
        sx = np.mean(np.linalg.norm(X_centered, axis=1))
        sy = np.mean(np.linalg.norm(Y_centered, axis=1))
    
    c = sy / sx if sx > 1e-8 else 1.0

    # Translation
    t = centroid_Y - c * np.dot(R, centroid_X)

    # Transformation matrix
    T = np.zeros((X.shape[1] + 1, X.shape[1] + 1))
    T[:X.shape[1], :X.shape[1]] = c * R
    T[:X.shape[1], -1] = t
    T[-1, -1] = 1

    return T

def homogeneous(coordinates):
    homogeneous_coordinates = np.hstack((coordinates, np.ones((coordinates.shape[0], 1))))
    return homogeneous_coordinates

def compute_alignment_error(src_pts, tar_pts, transform):
    """Compute alignment error after applying transformation"""
    transformed = (transform @ homogeneous(src_pts).T)[:3].T
    residuals = np.linalg.norm(transformed - tar_pts, axis=1)
    return residuals

def SKU_RANSAC(src_pts, tar_pts, n_samples=100, n_hypotheses=512, inlier_threshold=0.1):
    """
    RANSAC-based alignment with improved sampling and validation.
    
    Parameters:
    src_pts: Source point cloud (N, 3)
    tar_pts: Target point cloud (N, 3)
    n_samples: Number of points to sample per hypothesis
    n_hypotheses: Number of RANSAC iterations
    inlier_threshold: Distance threshold for inlier counting
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Downsample for faster RANSAC if we have too many points
    max_points_for_ransac = 10000
    if len(src_pts) > max_points_for_ransac:
        print(f"Downsampling from {len(src_pts)} to {max_points_for_ransac} points for RANSAC...")
        indices = np.random.choice(len(src_pts), max_points_for_ransac, replace=False)
        src_pts_ransac = src_pts[indices]
        tar_pts_ransac = tar_pts[indices]
    else:
        src_pts_ransac = src_pts
        tar_pts_ransac = tar_pts
    
    # Ensure we don't sample more points than available
    n_samples = min(n_samples, len(src_pts_ransac))
    
    best_score = -np.inf
    best_transform = np.identity(4)
    best_inliers = 0
    
    print(f"Running RANSAC with {n_samples} points per hypothesis, {n_hypotheses} iterations...")
    print(f"Inlier threshold: {inlier_threshold}")
    
    for hid in tqdm(range(n_hypotheses), desc="Running Umeyama RANSAC"):
        # Sample points
        ids = np.random.choice(len(src_pts_ransac), n_samples, replace=False)
        s_mini = src_pts_ransac[ids]
        t_mini = tar_pts_ransac[ids]
        
        # Compute transformation hypothesis
        try:
            hypo = umeyama_alignment(s_mini, t_mini, use_median=(n_samples < 20))
        except np.linalg.LinAlgError:
            continue
        
        # Evaluate on downsampled points
        residuals = compute_alignment_error(src_pts_ransac, tar_pts_ransac, hypo)
        
        # Count inliers
        inliers = np.sum(residuals < inlier_threshold)
        median_error = np.median(residuals)
        
        # Score: prioritize inlier count significantly
        score = inliers * 1000 - median_error
        
        if score > best_score:
            best_score = score
            best_transform = hypo
            best_inliers = inliers
            best_median_error = median_error
    
    inlier_pct = 100 * best_inliers / len(src_pts_ransac)
    print(f"Best alignment: {best_inliers}/{len(src_pts_ransac)} inliers ({inlier_pct:.1f}%), median error: {best_median_error:.4f}")
    
    # Refine with inliers if we have enough
    if best_inliers > n_samples * 2:
        print(f"Refining transformation with inliers...")
        residuals = compute_alignment_error(src_pts_ransac, tar_pts_ransac, best_transform)
        inlier_mask = residuals < inlier_threshold
        n_inliers_refine = np.sum(inlier_mask)
        if n_inliers_refine > 10:
            print(f"  Using {n_inliers_refine} inliers for refinement...")
            best_transform = umeyama_alignment(src_pts_ransac[inlier_mask], tar_pts_ransac[inlier_mask])
    
    return best_transform

def align_pcd(source:np.array, target:np.array, icp=None, init_trans=None, mask=None, 
              return_trans=True, voxel_size=0.1, n_samples=100, n_hypotheses=512, 
              inlier_threshold=0.1, use_fpfh=False, skip_icp=False):
    """ 
    Align the scale of source to target using umeyama,
    then refine the alignment using ICP.
    """
    # Apply initial transformation if provided
    if init_trans is not None:
        source = trimesh.transformations.transform_points(source, init_trans)
    
    # Apply mask if provided (use same mask for both)
    if mask is not None:
        source_for_align = source[mask]
        target_for_align = target[mask]
    else:
        source_for_align = source
        target_for_align = target
    
    print(f"Aligning {len(source_for_align)} source points to {len(target_for_align)} target points")
    
    # Analyze point clouds
    stats = analyze_point_clouds(source_for_align, target_for_align)
    
    # Normalize both point clouds
    print("Normalizing point clouds to unit scale...")
    src_norm, src_center, src_scale = normalize_point_cloud(source_for_align)
    tgt_norm, tgt_center, tgt_scale = normalize_point_cloud(target_for_align)
    
    # Adjust inlier threshold for normalized space
    norm_inlier_threshold = inlier_threshold / max(src_scale, tgt_scale)
    print(f"Normalized inlier threshold: {norm_inlier_threshold:.6f}")
    
    #####################################
    # First step: Coarse registration
    #####################################
    if use_fpfh:
        print("\nUsing FPFH-based global registration...")
        Rt_norm = fpfh_registration(src_norm, tgt_norm, voxel_size=0.05)
    else:
        print("\nUsing RANSAC + Umeyama registration...")
        Rt_norm = SKU_RANSAC(src_norm, tgt_norm, 
                             n_samples=n_samples, 
                             n_hypotheses=n_hypotheses,
                             inlier_threshold=norm_inlier_threshold)
    
    # Convert back to original space
    Rt_step1 = denormalize_transformation(Rt_norm, src_center, src_scale, tgt_center, tgt_scale)
    
    # Apply first transformation to full source cloud
    source_step1 = trimesh.transformations.transform_points(source, Rt_step1)
    
    if skip_icp:
        print("Skipping ICP refinement (--skip_icp flag set)")
        transformation_s2t = Rt_step1
        transformed_source = source_step1
    else:
        #####################################
        # Second step: Fine registration using ICP
        #####################################
        print("\nRunning point-to-plane ICP refinement...")
        icp_thr = voxel_size * 2

        # Downsample for ICP efficiency
        pcd_source_step1 = o3d.geometry.PointCloud()
        pcd_source_step1.points = o3d.utility.Vector3dVector(source_step1)
        pcd_source_step1 = pcd_source_step1.voxel_down_sample(voxel_size=voxel_size)
        
        pcd_target = o3d.geometry.PointCloud()
        pcd_target.points = o3d.utility.Vector3dVector(target)
        pcd_target = pcd_target.voxel_down_sample(voxel_size=voxel_size)
        pcd_target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # Choose ICP method
        if icp == "point":
            icp_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        elif icp == 'plain':
            icp_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        else:
            raise ValueError(f"Unknown ICP method: {icp}")
        
        # Run ICP
        reg_p2l = o3d.pipelines.registration.registration_icp(
            pcd_source_step1, pcd_target, icp_thr, np.identity(4), icp_method,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
        
        Rt_step2 = reg_p2l.transformation
        
        print(f"ICP converged: fitness={reg_p2l.fitness:.4f}, inlier_rmse={reg_p2l.inlier_rmse:.4f}")
        
        # Combine transformations and apply to original source
        transformation_s2t = Rt_step2 @ Rt_step1
        transformed_source = trimesh.transformations.transform_points(source, transformation_s2t)
    
    if return_trans:
        return transformed_source, transformation_s2t
    else:
        return transformed_source

if __name__ == "__main__":
    """
    The script consists of two parts:
    1. Align the predicted point cloud with the ground truth point cloud using the Umeyama and ICP algorithms. 
    2. calculate the reconstruction metrics.
    """
    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    num_sample_points = 100000
    voxelize_size = 0.005
    
    print("Loading point clouds...")
    gt_pcd = load_and_sample_gt_mesh(args.gt_pcd)
    print(f"GT point cloud: {gt_pcd.shape}")
    
    pred_pcd = load_pred_pcd_from_glb(args.pred_pcd)
    print(f"Predicted point cloud: {pred_pcd.shape}")
   
    # Perform alignment
    print("\n" + "="*60)
    print("Starting alignment process...")
    print("="*60)
    
    aligned_pcd, trans = align_pcd(
        pred_pcd, gt_pcd, 
        init_trans=None, 
        mask=None,
        icp=args.icp, 
        return_trans=True,
        voxel_size=voxelize_size*2,
        n_samples=args.n_samples,
        n_hypotheses=args.n_hypotheses,
        inlier_threshold=args.inlier_threshold,
        use_fpfh=args.use_fpfh,
        skip_icp=args.skip_icp
    )
    
    print("\n" + "="*60)
    print("Alignment complete!")
    print("="*60)
    
    # Compute final alignment error using KDTree (memory efficient)
    print("\nComputing alignment metrics...")
    tree = KDTree(gt_pcd)
    distances, _ = tree.query(aligned_pcd, k=1)
    
    print(f"\nFinal alignment metrics:")
    print(f"  Mean distance: {np.mean(distances):.6f}")
    print(f"  Median distance: {np.median(distances):.6f}")
    print(f"  90th percentile: {np.percentile(distances, 90):.6f}")
    print(f"  95th percentile: {np.percentile(distances, 95):.6f}")
    print(f"  99th percentile: {np.percentile(distances, 99):.6f}")

    # ----- Visualization -----
    print("\nPreparing visualization...")

    # GT (Green)
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(gt_pcd)
    pcd_gt.paint_uniform_color([0, 1, 0])

    # Original Pred (Blue)
    pcd_pred_raw = o3d.geometry.PointCloud()
    pcd_pred_raw.points = o3d.utility.Vector3dVector(pred_pcd)
    pcd_pred_raw.paint_uniform_color([0, 0, 1])

    # Aligned Pred (Red)
    pcd_pred_aligned = o3d.geometry.PointCloud()
    pcd_pred_aligned.points = o3d.utility.Vector3dVector(aligned_pcd)
    pcd_pred_aligned.paint_uniform_color([1, 0, 0])

    print("\nVisualizing: GT (Green) | Original Pred (Blue) | Aligned Pred (Red)")
    print("Close the window to exit...")

    o3d.visualization.draw_geometries(
        [pcd_gt, pcd_pred_raw, pcd_pred_aligned],
        window_name="GT (Green) | Original (Blue) | Aligned (Red)",
        width=1400,
        height=900,
        point_show_normal=False
    )