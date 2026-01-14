import open3d as o3d
import numpy as np
import argparse
import sys
import os

def load_pcd(path):
    if not os.path.exists(path):
        print(f"Error: File not found: {path}")
        sys.exit(1)
    
    try:
        pcd = o3d.io.read_point_cloud(path)
        if not pcd.has_points():
            print(f"Warning: {path} contains no points.")
        return pcd
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        sys.exit(1)


def visualize_single(file_path):
    print(f"Visualizing single file: {file_path}")
    pcd = load_pcd(file_path)
    
    # If no colors exist, paint it with a nice coordinate-based gradient or simple black
    if not pcd.has_colors():
        pcd.paint_uniform_color([0, 0, 0])  # Black points
    
    # Create visualizer with white background for better contrast
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Viewer: {os.path.basename(file_path)}", width=1024, height=768)
    vis.add_geometry(pcd)
    
    # Optional: Add coordinate frame
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0]))
    
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1]) # White background
    opt.point_size = 2.0
    
    vis.run()
    vis.destroy_window()


def visualize_compare(original_path, augmented_path):
    print(f"Comparing:")
    print(f"  Original (Gray): {original_path}")
    print(f"  Augmented (Red): {augmented_path}")
    
    pcd_orig = load_pcd(original_path)
    pcd_aug = load_pcd(augmented_path)
    
    # Paint Original -> Gray
    pcd_orig.paint_uniform_color([0.1, 0.1, 0.1])
    
    # Paint Augmented -> Red
    pcd_aug.paint_uniform_color([1.0, 0.0, 0.0])
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Comparison Mode (Gray=Orig, Red=Aug)", width=1024, height=768)
    
    vis.add_geometry(pcd_orig)
    vis.add_geometry(pcd_aug)
    
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])
    opt.point_size = 2.0
    
    vis.run()
    vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description="LiDAR Point Cloud Visualizer")
    parser.add_argument("file1", help="Path to the first .pcd file")
    parser.add_argument("file2", nargs="?", help="(Optional) Path to a second .pcd file for comparison")
    
    args = parser.parse_args()
    
    if args.file2:
        visualize_compare(args.file1, args.file2)
    else:
        visualize_single(args.file1)

if __name__ == "__main__":
    main()
