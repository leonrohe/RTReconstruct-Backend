import open3d as o3d
import numpy as np
import argparse
import os


def compute_metrics(gt_mesh_path, recon_mesh_path, num_samples=500000, threshold=0.01):
    # Load meshes
    gt_mesh = o3d.io.read_triangle_mesh(gt_mesh_path)
    recon_mesh = o3d.io.read_triangle_mesh(recon_mesh_path)
    # rotate recon model -90, 0, 180 degrees
    R = recon_mesh.get_rotation_matrix_from_xyz((-np.pi/2, 0, np.pi))
    recon_mesh.rotate(R, center=(0, 0, 0))

    if not gt_mesh.has_triangles() or not recon_mesh.has_triangles():
        raise ValueError("Both meshes must have faces/triangles.")

    # Align coordinate systems
    # gt_mesh = align_unity_to_open3d(gt_mesh)
    # recon_mesh = align_unity_to_open3d(recon_mesh)

    # Normalize scale if needed (optional)
    # gt_mesh.scale(1.0 / np.max(gt_mesh.get_max_bound() - gt_mesh.get_min_bound()), center=gt_mesh.get_center())
    # recon_mesh.scale(1.0 / np.max(recon_mesh.get_max_bound() - recon_mesh.get_min_bound()), center=recon_mesh.get_center())

    # Sample points
    print("Sampling points from meshes...")
    gt_pcd = gt_mesh.sample_points_poisson_disk(num_samples)
    recon_pcd = recon_mesh.sample_points_poisson_disk(num_samples)

    # Compute distances both ways
    print("Computing distances...")
    dists_gt_to_recon = np.asarray(gt_pcd.compute_point_cloud_distance(recon_pcd))
    dists_recon_to_gt = np.asarray(recon_pcd.compute_point_cloud_distance(gt_pcd))

    # Chamfer Distance
    chamfer_dist = (dists_gt_to_recon.mean() + dists_recon_to_gt.mean()) / 2.0

    # Accuracy (mean recon→GT distance)
    accuracy = dists_recon_to_gt.mean()

    # Completeness (mean GT→recon distance)
    completeness = dists_gt_to_recon.mean()

    # Precision, Recall (Completeness%), F-score
    prec = np.mean(dists_recon_to_gt < threshold)
    rec = np.mean(dists_gt_to_recon < threshold)
    fscore = 2 * prec * rec / (prec + rec + 1e-8)

    print(f"\n📊 Results")
    print(f"Chamfer Distance: {chamfer_dist*1000:.3f} mm")
    print(f"Accuracy (Recon→GT): {accuracy*1000:.3f} mm")
    print(f"Completeness (GT→Recon): {completeness*1000:.3f} mm")
    print(f"Precision (<{threshold*100:.1f} cm): {prec*100:.2f}%")
    print(f"Recall / Completeness% (<{threshold*100:.1f} cm): {rec*100:.2f}%")
    print(f"F-score: {fscore*100:.2f}%")

    # Create colored visualization
    print("Creating error heatmap...")
    colors = np.zeros((len(dists_recon_to_gt), 3))
    max_err = np.percentile(dists_recon_to_gt, 99)  # clamp outliers
    for i, d in enumerate(dists_recon_to_gt):
        t = min(d / max_err, 1.0)
        colors[i] = [t, 0, 1 - t]  # red = high error, blue = low error
    recon_pcd.colors = o3d.utility.Vector3dVector(colors)

    # Save and show
    out_file = os.path.splitext(recon_mesh_path)[0] + "_error_vis.ply"
    o3d.io.write_point_cloud(out_file, recon_pcd)
    print(f"Saved colored error visualization to: {out_file}")

    o3d.visualization.draw_geometries([recon_pcd], window_name="Reconstruction Error Map")

    return chamfer_dist, fscore


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate reconstruction accuracy against Unity ground truth.")
    parser.add_argument("--gt", required=True, help="Path to Unity ground-truth mesh (.obj/.ply)")
    parser.add_argument("--recon", required=True, help="Path to reconstructed mesh (.obj/.ply)")
    parser.add_argument("--threshold", type=float, default=0.01, help="F-score distance threshold in meters (default=0.01m=1cm)")
    args = parser.parse_args()

    compute_metrics(args.gt, args.recon, threshold=args.threshold)
