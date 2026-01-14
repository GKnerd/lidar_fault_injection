import logging
import json
import numpy as np
import open3d as o3d

from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Ensure lidar_augmenter.py is in the same folder
from .lidar_corruptions import LidarAugmenter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LidarInjector")

class LidarInjector:
    """
    The User-Facing Builder Class.
    Allows users to define experiments using Python code.
    """
    def __init__(self, input_dir, output_dir):
        # Get current time: YYYY-MM-DD_HH-MM-SS
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_output = Path(output_dir)
        timestamped_output_dir = base_output.parent / f"{base_output.name}_{timestamp}"

        # Update the config with this new unique path
        self.config = {
            "io": {
                "input_dir": input_dir,
                "output_dir": str(timestamped_output_dir),
                "num_workers": 4
            },
            "sampling": {
                "mode": "count",
                "value": 100, # Default
                "seed": 42
            },
            "augmentation_groups": []
        }
        self.current_group = None


    def set_sampling(self, mode="count", value=100, seed=42):
        """
        Configure how many files to process.
        :param mode: "count" (exact number) or "ratio" (percentage 0.0-1.0)
        :param value: The number or ratio.
        """
        self.config["sampling"] = {"mode": mode, "value": value, "seed": seed}
        return self


    def set_workers(self, num_workers):
        self.config["io"]["num_workers"] = num_workers
        return self


    def add_group(self, name, fraction):
        """
        Start defining a new augmentation group.
        :param fraction: Percentage of the sampled data this group should occupy (0.0-1.0).
        """
        self.current_group = {
            "name": name,
            "fraction": fraction,
            "pipeline": []
        }
        self.config["augmentation_groups"].append(self.current_group)
        return self


    def apply(self, corruption_name, **kwargs):
        """
        Add a corruption to the current group.
        Example: .apply("Sunlight Glare", severity=3)
        """
        if self.current_group is None:
            raise ValueError("You must call .add_group() before calling .apply()!")
        
        # Add entry to pipeline
        step = {"corruption": corruption_name}
        step.update(kwargs) # Merge parameters like 'severity' or 'rate'
        
        self.current_group["pipeline"].append(step)
        return self


    def run(self):
        """
        Compiles the configuration and executes the pipeline.
        """
        # Instantiate the worker with the generated config
        pipeline = InternalPipeline(self.config)
        pipeline.run()

# ==============================================================================
# INTERNAL WORKER LOGIC 
# ==============================================================================

class InternalPipeline:
    
    def __init__(self, config):
        self.config = config
        self.input_dir = Path(config['io']['input_dir'])
        self.output_dir = Path(config['io']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.augmenter = LidarAugmenter(seed=config['sampling'].get('seed', 42))

        # Mapping "Nice Names" to Internal Method Names
        self.corruption_mapping = {
            "Sunlight Glare":                   "scene_glare_noise",
            "Crosstalk":                        "lidar_crosstalk_noise",
            "Impulse Noise":                    "impulse_noise",
            "Random Dropout":                   "random_dropout",
            "Structured Dropout":               "structured_dropout",
            "Motion Distortion":                "simulate_motion_distortion",
            "Occlusion":                        "simulate_occlusion",
            "Gaussian Noise":                   "add_gaussian_noise",
            "Field of View (FOV) Reduction":    "reduce_fov",
            "Sparse Scan":                      "sparse_scan_pattern"
        }


    def _load_pcd(self, path):
        try:
            pcd = o3d.io.read_point_cloud(str(path))
            points = np.asarray(pcd.points)
            return points if len(points) > 0 else None
        except:
            return None


    def _save_pcd(self, points, path):
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(str(path), pcd)


    def _apply_pipeline_steps(self, points, pipeline_config):
        current_points = points.copy()
        log = []

        for step in pipeline_config:
            params = step.copy()
            nice_name = params.pop("corruption")
            
            if nice_name not in self.corruption_mapping:
                logger.warning(f"Unknown corruption: {nice_name}")
                continue

            method_name = self.corruption_mapping[nice_name]
            
            # Parameter restructuring for complex methods
            func_kwargs = params # Default: flat params match arguments
            
            if nice_name == "Motion Distortion":
                func_kwargs = {
                    "timestamps": np.linspace(0, 1e8, len(current_points)), # Mock timestamps
                    "motion_params": {
                        "linear_velocity": params.get("linear_velocity", [0,0,0]),
                        "angular_velocity": params.get("angular_velocity", [0,0,0])
                    }
                }
            elif nice_name == "Occlusion":
                func_kwargs = {"occlusion_params": params}
            elif nice_name == "Gaussian Noise":
                func_kwargs = {"noise_params": params}
            elif nice_name == "Field of View (FOV) Reduction":
                func_kwargs = {"fov_reduction": params}

            # Execute
            if hasattr(self.augmenter, method_name):
                try:
                    res = getattr(self.augmenter, method_name)(current_points, **func_kwargs)
                    current_points = res[0] if isinstance(res, tuple) else res
                    log.append({"corruption": nice_name, "params": params})
                except Exception as e:
                    logger.error(f"Failed to apply {nice_name}: {e}")

        return current_points, log


    def process_task(self, task):
        # Unpack task
        file_path = task['file_path']
        pipeline = task['pipeline']
        group_name = task['group_name']

        points = self._load_pcd(file_path)
        if points is None: return None

        aug_points, meta = self._apply_pipeline_steps(points, pipeline)

        # Naming convention: original_Group_Name.pcd
        safe_name = group_name.replace(" ", "_")
        new_name = f"{file_path.stem}_{safe_name}{file_path.suffix}"
        save_path = self.output_dir / new_name

        self._save_pcd(aug_points, save_path)

        return {
            "original": str(file_path.name),
            "augmented": new_name,
            "group": group_name,
            "steps": meta
        }


    def run(self):
        # Scan and Sample Files
        all_files = list(self.input_dir.glob("*.pcd"))
        total_files = len(all_files)
        if total_files == 0:
            logger.error("No input files found!")
            return

        # Sampling logic
        s_conf = self.config['sampling']
        count = int(s_conf['value']) if s_conf['mode'] == 'count' else int(total_files * s_conf['value'])
        count = min(count, total_files)
        
        rng = np.random.default_rng(s_conf['seed'])
        selected_files = rng.choice(all_files, count, replace=False)
        
        logger.info(f"Selected {count} files for processing.")

        # Assign Files to Groups
        tasks = []
        start_idx = 0
        
        for group in self.config['augmentation_groups']:
            grp_count = int(count * group['fraction'])
            end_idx = start_idx + grp_count
            grp_files = selected_files[start_idx:end_idx]
            
            for f in grp_files:
                tasks.append({
                    "file_path": f,
                    "pipeline": group['pipeline'],
                    "group_name": group['name']
                })
            start_idx = end_idx

        # Parallel Execution
        results = []
        workers = self.config['io']['num_workers']
        logger.info(f"Processing {len(tasks)} augmentations with {workers} workers...")
        
        with ProcessPoolExecutor(max_workers=workers) as exe:
            # We map a helper method to unpack the task dict
            futures = list(tqdm(exe.map(self.process_task, tasks), total=len(tasks)))
            results = [r for r in futures if r is not None]

        # Save Metadata
        meta_file = self.output_dir / "metadata.json"
        with open(meta_file, 'w') as f:
            json.dump({r['augmented']: r for r in results}, f, indent=4)
            
        logger.info(f"Done! Metadata saved to {meta_file}")
