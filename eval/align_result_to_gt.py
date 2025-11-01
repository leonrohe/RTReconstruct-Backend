# --------------------------------------------------------
# Optimized FPFH-based Point Cloud Alignment
# --------------------------------------------------------
import os
import argparse
import numpy as np
import trimesh
import open3d as o3d
from scipy.spatial import cKDTree as KDTree

parser = argparse.ArgumentParser(description="FPFH-based point cloud alignment")
parser.add_argument("--gt_pcd", type=str, required=True, help='Ground truth point cloud path')
parser.add_argument("--pred_pcd", type=str, required=True, help='Predicted point cloud path')
parser.add_argument("--voxel_size", type=float, default=0.05, help='Voxel size for downsampling (default: 0.05)')
parser.add_argument("--icp", type=str, default='point', choices=['point', 'plane'], help='ICP method')
parser.add_argument("--icp_threshold", type=float, default=0.02, help='ICP distance threshold')
parser.add_argument("--max_icp_iterations", type=int, default=100, help='Maximum ICP iterations')
parser.add_argument("--fpfh_radius_multiplier", type=float, default=5.0, help='FPFH feature radius = voxel_size * this')
parser.add_argument("--ransac_max_iterations", type=int, default=100000, help='RANSAC max iterations')
parser.add_argument("--ransac_confidence", type=float, default=0.999, help='RANSAC confidence')
parser.add_argument("--skip_icp", action="store_true", help='Skip ICP refinement')
parser.add_argument("--save_aligned", type=str, default=None, help='Save aligned point cloud to this path')
parser.add_argument("--visualize", action="store_true", help='Visualize results')

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
    """Analyze and display point cloud statistics"""
    print("\n" + "="*70)
    print("POINT CLOUD ANALYSIS")
    print("="*70)
    
    def print_stats(name, pcd):
        extent = pcd.max(axis=0) - pcd.min(axis=0)
        diagonal = np.linalg.norm(extent)
        print(f"\n{name}:")
        print(f"  Points:   {len(pcd):,}")
        print(f"  Center:   [{pcd.mean(axis=0)[0]:8.4f}, {pcd.mean(axis=0)[1]:8.4f}, {pcd.mean(axis=0)[2]:8.4f}]")
        print(f"  Extent:   [{extent[0]:8.4f}, {extent[1]:8.4f}, {extent[2]:8.4f}]")
        print(f"  Diagonal: {diagonal:8.4f}")
        return diagonal
    
    src_diag = print_stats("Source (Predicted)", src)
    tgt_diag = print_stats("Target (Ground Truth)", tgt)
    
    scale_ratio = tgt_diag / src_diag
    center_dist = np.linalg.norm(src.mean(axis=0) - tgt.mean(axis=0))
    
    print(f"\nRelative Metrics:")
    print(f"  Scale ratio (GT/Pred): {scale_ratio:.4f}")
    print(f"  Center distance:       {center_dist:.4f}")
    print("="*70)
    
    return scale_ratio, center_dist

def preprocess_point_cloud(points, voxel_size):
    """Downsample and estimate normals"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Downsample
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    # Estimate normals
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    return pcd_down

def compute_fpfh_features(pcd, voxel_size, radius_multiplier=5.0):
    """Compute FPFH features for point cloud"""
    radius_feature = voxel_size * radius_multiplier
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return fpfh

def normalize_point_cloud(pcd):
    """Normalize point cloud to unit sphere centered at origin"""
    center = pcd.mean(axis=0)
    pcd_centered = pcd - center
    scale = np.max(np.linalg.norm(pcd_centered, axis=1))
    pcd_normalized = pcd_centered / scale
    return pcd_normalized, center, scale

def denormalize_transformation(T, src_center, src_scale, tgt_center, tgt_scale):
    """Convert transformation from normalized space back to original space"""
    # Build normalization transformations
    T_normalize_src = np.eye(4)
    T_normalize_src[:3, 3] = -src_center
    T_scale_src = np.eye(4)
    T_scale_src[:3, :3] = np.eye(3) / src_scale
    
    # Build denormalization transformations
    T_scale_tgt = np.eye(4)
    T_scale_tgt[:3, :3] = np.eye(3) * tgt_scale
    T_denormalize_tgt = np.eye(4)
    T_denormalize_tgt[:3, 3] = tgt_center
    
    # Compose: denormalize_tgt @ scale_tgt @ T @ scale_src @ normalize_src
    T_full = T_denormalize_tgt @ T_scale_tgt @ T @ T_scale_src @ T_normalize_src
    
    return T_full

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, 
                                voxel_size, max_iterations=100000, confidence=0.999):
    """Execute RANSAC-based global registration using FPFH features"""
    distance_threshold = voxel_size * 1.5
    
    print(f"\nGlobal Registration (FPFH + RANSAC):")
    print(f"  Distance threshold: {distance_threshold:.4f}")
    print(f"  Max iterations:     {max_iterations:,}")
    print(f"  Confidence:         {confidence}")
    
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, 
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,  # Use 4 points instead of 3 for better stability
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iterations, confidence))
    
    print(f"\n  Result:")
    print(f"    Fitness:      {result.fitness:.4f}")
    print(f"    Inlier RMSE:  {result.inlier_rmse:.4f}")
    print(f"    Correspondences: {len(result.correspondence_set)}")
    
    return result

def refine_registration_icp(source, target, init_transform, threshold, method='point', max_iterations=100):
    """Refine alignment using ICP"""
    print(f"\nICP Refinement ({method}-to-{method}):")
    print(f"  Threshold:      {threshold:.4f}")
    print(f"  Max iterations: {max_iterations}")
    
    if method == 'point':
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    else:  # plane
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    
    result = o3d.pipelines.registration.registration_icp(
        source, target, threshold, init_transform, estimation,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations))
    
    print(f"\n  Result:")
    print(f"    Fitness:      {result.fitness:.4f}")
    print(f"    Inlier RMSE:  {result.inlier_rmse:.4f}")
    
    return result

def compute_alignment_metrics(aligned_pcd, gt_pcd):
    """Compute comprehensive alignment metrics"""
    print("\nComputing alignment metrics...")
    
    # Use KDTree for efficient nearest neighbor search
    tree = KDTree(gt_pcd)
    distances, _ = tree.query(aligned_pcd, k=1)
    
    metrics = {
        'mean': np.mean(distances),
        'median': np.median(distances),
        'std': np.std(distances),
        'min': np.min(distances),
        'max': np.max(distances),
        'p90': np.percentile(distances, 90),
        'p95': np.percentile(distances, 95),
        'p99': np.percentile(distances, 99),
    }
    
    # Count points within various thresholds
    thresholds = [0.01, 0.05, 0.1, 0.5]
    for thresh in thresholds:
        pct = 100 * np.sum(distances < thresh) / len(distances)
        metrics[f'within_{thresh}'] = pct
    
    return metrics, distances

def print_metrics(metrics):
    """Pretty print alignment metrics"""
    print("\n" + "="*70)
    print("ALIGNMENT METRICS")
    print("="*70)
    print(f"\nDistance Statistics:")
    print(f"  Mean:     {metrics['mean']:.6f}")
    print(f"  Median:   {metrics['median']:.6f}")
    print(f"  Std Dev:  {metrics['std']:.6f}")
    print(f"  Min:      {metrics['min']:.6f}")
    print(f"  Max:      {metrics['max']:.6f}")
    
    print(f"\nPercentiles:")
    print(f"  90th:     {metrics['p90']:.6f}")
    print(f"  95th:     {metrics['p95']:.6f}")
    print(f"  99th:     {metrics['p99']:.6f}")
    
    print(f"\nPoints Within Threshold:")
    print(f"  < 0.01m:  {metrics['within_0.01']:.2f}%")
    print(f"  < 0.05m:  {metrics['within_0.05']:.2f}%")
    print(f"  < 0.1m:   {metrics['within_0.1']:.2f}%")
    print(f"  < 0.5m:   {metrics['within_0.5']:.2f}%")
    print("="*70)

def visualize_alignment(pred_original, pred_aligned, gt_pcd):
    """Visualize alignment results"""
    print("\nPreparing visualization...")
    
    # Ground truth (Green)
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(gt_pcd)
    pcd_gt.paint_uniform_color([0.0, 0.8, 0.0])  # Green
    
    # Original prediction (Blue)
    pcd_pred_raw = o3d.geometry.PointCloud()
    pcd_pred_raw.points = o3d.utility.Vector3dVector(pred_original)
    pcd_pred_raw.paint_uniform_color([0.0, 0.4, 1.0])  # Blue
    
    # Aligned prediction (Red)
    pcd_pred_aligned = o3d.geometry.PointCloud()
    pcd_pred_aligned.points = o3d.utility.Vector3dVector(pred_aligned)
    pcd_pred_aligned.paint_uniform_color([1.0, 0.0, 0.0])  # Red
    
    print("\nVisualization Legend:")
    print("  Green = Ground Truth")
    print("  Blue  = Original Prediction")
    print("  Red   = Aligned Prediction")
    print("\nClose window to continue...")
    
    o3d.visualization.draw_geometries(
        [pcd_gt, pcd_pred_raw, pcd_pred_aligned],
        window_name="Alignment Result: GT (Green) | Original (Blue) | Aligned (Red)",
        width=1600,
        height=1000,
        point_show_normal=False
    )

def main():
    args = parser.parse_args()
    
    print("="*70)
    print("FPFH-BASED POINT CLOUD ALIGNMENT")
    print("="*70)
    
    # Load point clouds
    print("\nLoading point clouds...")
    pred_pcd = load_pred_pcd_from_glb(args.pred_pcd)
    print(f"  Predicted: {pred_pcd.shape[0]:,} points")
    
    gt_pcd = load_and_sample_gt_mesh(args.gt_pcd)
    print(f"  Ground Truth: {gt_pcd.shape[0]:,} points")
    
    # Analyze point clouds
    scale_ratio, center_dist = analyze_point_clouds(pred_pcd, gt_pcd)
    
    # Normalize point clouds to unit scale (critical for FPFH to work correctly)
    print("\n" + "="*70)
    print("NORMALIZATION")
    print("="*70)
    print("\nNormalizing point clouds to unit scale...")
    
    pred_norm, pred_center, pred_scale = normalize_point_cloud(pred_pcd)
    print(f"  Source: center={pred_center}, scale={pred_scale:.4f}")
    
    gt_norm, gt_center, gt_scale = normalize_point_cloud(gt_pcd)
    print(f"  Target: center={gt_center}, scale={gt_scale:.4f}")
    
    # Preprocess
    print("\n" + "="*70)
    print("PREPROCESSING")
    print("="*70)
    print(f"\nVoxel size: {args.voxel_size}")
    
    print("\nDownsampling and computing normals...")
    source_down = preprocess_point_cloud(pred_norm, args.voxel_size)
    print(f"  Source downsampled: {len(source_down.points):,} points")
    
    target_down = preprocess_point_cloud(gt_norm, args.voxel_size)
    print(f"  Target downsampled: {len(target_down.points):,} points")
    
    # Compute FPFH features
    print("\nComputing FPFH features...")
    source_fpfh = compute_fpfh_features(source_down, args.voxel_size, args.fpfh_radius_multiplier)
    target_fpfh = compute_fpfh_features(target_down, args.voxel_size, args.fpfh_radius_multiplier)
    print(f"  Feature dimension: {source_fpfh.data.shape[0]}")
    
    # Global registration
    print("\n" + "="*70)
    print("GLOBAL REGISTRATION")
    print("="*70)
    
    result_ransac = execute_global_registration(
        source_down, target_down, source_fpfh, target_fpfh,
        args.voxel_size, args.ransac_max_iterations, args.ransac_confidence
    )
    
    # Transform from normalized space back to original space
    transformation_normalized = result_ransac.transformation
    transformation = denormalize_transformation(
        transformation_normalized, pred_center, pred_scale, gt_center, gt_scale
    )
    
    print(f"\nTransformation matrix (in original space):")
    print(transformation)
    
    # ICP refinement
    if not args.skip_icp:
        print("\n" + "="*70)
        print("ICP REFINEMENT")
        print("="*70)
        
        # Apply initial transformation
        source_temp = o3d.geometry.PointCloud()
        source_temp.points = o3d.utility.Vector3dVector(pred_pcd)
        source_temp.transform(transformation)
        
        # Downsample for ICP
        source_temp = source_temp.voxel_down_sample(args.voxel_size)
        target_temp = o3d.geometry.PointCloud()
        target_temp.points = o3d.utility.Vector3dVector(gt_pcd)
        target_temp = target_temp.voxel_down_sample(args.voxel_size)
        
        # Estimate normals for plane-to-plane ICP
        if args.icp == 'plane':
            source_temp.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size * 2, max_nn=30))
            target_temp.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size * 2, max_nn=30))
        
        result_icp = refine_registration_icp(
            source_temp, target_temp, np.identity(4),
            args.icp_threshold, args.icp, args.max_icp_iterations
        )
        
        transformation = result_icp.transformation @ transformation
    
    # Apply final transformation
    print("\n" + "="*70)
    print("APPLYING TRANSFORMATION")
    print("="*70)
    
    aligned_pcd = trimesh.transformations.transform_points(pred_pcd, transformation)
    print(f"\nTransformed {len(aligned_pcd):,} points")
    
    # Compute metrics
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)
    
    metrics, distances = compute_alignment_metrics(aligned_pcd, gt_pcd)
    print_metrics(metrics)
    
    # Save aligned point cloud
    if args.save_aligned:
        print(f"\nSaving aligned point cloud to: {args.save_aligned}")
        pcd_save = o3d.geometry.PointCloud()
        pcd_save.points = o3d.utility.Vector3dVector(aligned_pcd)
        o3d.io.write_point_cloud(args.save_aligned, pcd_save)
        print("  Saved successfully!")
    
    # Visualize
    if args.visualize:
        visualize_alignment(pred_pcd, aligned_pcd, gt_pcd)
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70 + "\n")
    
    return metrics

if __name__ == "__main__":
    main()