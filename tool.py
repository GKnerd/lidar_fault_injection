import os
import json
import logging
import numpy as np
import open3d as o3d

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from functools import partial

from lidar_corruptions import LidarAugmenter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Lidar Augmenter")

class AugmentationPipeline:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.input_dir = Path(self.config['IO']['input_dir'])
        
        self.output_dir = Path(self.config['IO']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Augmenter with seed from config
        self.augmenter = LidarAugmenter(seed=self.config['sampling'].get('seed', 42))

    def _load_pcd_as_numpy(self, file_path):
        """
        Lazy loader: Reads a single .pcd file into (N, 3) numpy array.
        Modify this if your PCDs contain intensity (N, 4).
        """
        try:
            pcd = o3d.io.read_point_cloud(str(file_path))
            # Open3D loads as Vector3dVector, convert to float64 numpy
            points = np.asarray(pcd.points)
            if len(points) == 0:
                return None
            return points
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return None

    def _save_numpy_as_pcd(self, points, save_path):
        """
        Saves (N, 3) numpy array back to .pcd format.
        """
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            o3d.io.write_point_cloud(str(save_path), pcd)
        except Exception as e:
            logger.error(f"Failed to save {save_path}: {e}")

    def _apply_pipeline(self, points):
        """
        Sequentially applies the list of corruptions defined in config.
        """
        meta_log = []
        current_points = points.copy()

        for step in self.config['pipeline']:
            method_name = step['corruption']
            params = step.get('params', {})
            
            # Dynamically call the method from LidarAugmenter
            if hasattr(self.augmenter, method_name):
                func = getattr(self.augmenter, method_name)
                
                # Handle different return types (some return mask, some just cloud)
                # We inspect the method signature or simply check return type
                try:
                    result = func(current_points, **params)
                    
                    # If function returns tuple (cloud, mask), take cloud
                    if isinstance(result, tuple):
                        current_points = result[0]
                    else:
                        current_points = result
                    
                    meta_log.append({
                        "corruption": method_name,
                        "params": params
                    })
                except Exception as e:
                    logger.error(f"Error applying {method_name}: {e}")
            else:
                logger.warning(f"Method {method_name} not found in LidarAugmenter")

        return current_points, meta_log

    def process_single_file(self, file_path):
        """
        Worker function: Loads -> Augments -> Saves.
        Returns metadata entry.
        """
        # Load
        points = self._load_pcd_as_numpy(file_path)
        if points is None:
            return None

        # Augment
        aug_points, operations = self._apply_pipeline(points)

        # Generate new filename
        # e.g., "scan_123.pcd" -> "scan_123_aug.pcd"
        new_filename = file_path.stem + "_aug" + file_path.suffix
        save_path = self.output_dir / new_filename

        # Save
        self._save_numpy_as_pcd(aug_points, save_path)

        # Return metadata
        return {
            "original_file": str(file_path.name),
            "augmented_file": new_filename,
            "augmentations": operations
        }

    def get_target_files(self):
        """
        Scans directory and selects files based on sampling config.
        Uses generators to avoid loading millions of strings if possible,
        though random sampling requires a list or reservoir sampling.
        """
        logger.info("Scanning directory...")
        # Glob returns a generator
        all_files_gen = self.input_dir.glob("*.pcd")
        
        # For random sampling, we unfortunately need the list or the total count.
        # With "millions", converting to list takes ~1-5 seconds and ~100MB RAM.
        # This is acceptable for most modern machines.
        all_files = list(all_files_gen) 
        total_files = len(all_files)
        
        if total_files == 0:
            raise ValueError(f"No .pcd files found in {self.input_dir}")

        mode = self.config['sampling']['mode']
        val = self.config['sampling']['value']
        rng = np.random.default_rng(self.config['sampling'].get('seed', 42))

        if mode == 'count':
            num_samples = min(int(val), total_files)
        elif mode == 'ratio':
            num_samples = int(total_files * np.clip(val, 0.0, 1.0))
        else:
            num_samples = total_files

        logger.info(f"Selected {num_samples} files out of {total_files} for augmentation.")
        
        # Randomly select files
        return rng.choice(all_files, num_samples, replace=False)

    def run(self):
        files_to_process = self.get_target_files()
        metadata = {}
        max_workers = self.config['IO'].get('num_workers', 4)

        logger.info(f"Starting processing with {max_workers} workers...")
        
        # Parallel Processing Loop
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # list() forces execution so tqdm works
            results = list(tqdm(
                executor.map(self.process_single_file, files_to_process), 
                total=len(files_to_process),
                unit="pcd"
            ))

        # Filter out failed processes (None results)
        valid_results = [r for r in results if r is not None]
        
        # Compile Metadata
        for res in valid_results:
            metadata[res['original_file']] = res

        # Save Metadata.json
        meta_path = self.output_dir / "metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        logger.info(f"Processing complete. Metadata saved to {meta_path}")

if __name__ == "__main__":
    # Ensure config.json exists
    if not os.path.exists("config.json"):
        print("Error: config.json not found.")
    else:
        pipeline = AugmentationPipeline("config.json")
        pipeline.run()
