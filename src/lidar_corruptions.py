import numpy as np
import logging

from scipy.spatial.transform import Rotation as R
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LidarAugmenter")

class LidarAugmenter:
    """
    LidarAugmenter class designed to to artificially corrupt or "augment" LiDAR point cloud data.

    """
    def __init__(self, seed=None):
        """
        Initialize the Augmenter.
        :param seed: Random seed for reproducibility.
        """
        if seed is None:
            seed = np.random.randint(0, 10000)
        self.rng = np.random.default_rng(seed)         
        logger.debug("LiDAR Augmenter initialized")

    # =============================================================================
    # ENVIRONMENTAL INTERFERENCE 
    # =============================================================================
    """
    taken from https://github.com/thu-ml/3D_Corruptions_AD (last access: 11.01.2026)
    """
    def scene_glare_noise(self, cloud, severity):
        """
        Simulates sunlight glare by adding high-variance noise to a random subset of points.
        This often occurs when the sensor faces bright light sources directly.

        It selects a small percentage of points (1%-5% based on severity) and adds very strong Gaussian noise 
        (multiplied by 2.0) to them. This creates a "blinding" effect where valid points are scattered chaotically.
        """
        if cloud is None or len(cloud) == 0:
            return cloud

        N, C = cloud.shape
        # Severity determines % of points affected (1% to 5%)
        rates = [0.010, 0.020, 0.030, 0.040, 0.050]
        # Ensure severity is within bounds (1-5)
        sev_idx = np.clip(severity - 1, 0, 4)
        c = int(rates[sev_idx] * N)

        if c == 0:
            return cloud

        idx = self.rng.choice(N, c, replace=False)
        
        # Glare is modeled as strong normal noise (factor 2.0)
        noise = self.rng.normal(size=(c, C)) * 2.0
        
        result_cloud = cloud.copy()
        result_cloud[idx] += noise
        
        return result_cloud


    # =============================================================================
    # SENSOR INTERFERENCE 
    # =============================================================================
    """
    taken from https://github.com/thu-ml/3D_Corruptions_AD (last access: 11.01.2026)
    """
    def lidar_crosstalk_noise(self, cloud, severity):
        """
        Simulates crosstalk: interference between laser emitters or returns from 
        different beams, resulting in noise clusters near valid points. 

        It selects a small percentage of points (1%-5% based on severity) and adds 
        very strong Gaussian noise (multiplied by 2.0) to them. 
        This creates a "blinding" effect where valid points are scattered chaotically.
        """
        if cloud is None or len(cloud) == 0:
            return cloud

        N, C = cloud.shape
        # Crosstalk affects a small percentage (0.4% to 2.0%)
        rates = [0.004, 0.008, 0.012, 0.016, 0.020]
        sev_idx = np.clip(severity - 1, 0, 4)
        c = int(rates[sev_idx] * N)

        if c == 0:
            return cloud

        idx = self.rng.choice(N, c, replace=False)

        # 

        # Crosstalk noise is typically stronger/spikier than standard Gaussian (factor 3.0)
        noise = self.rng.normal(size=(c, C)) * 3.0
        
        result_cloud = cloud.copy()
        result_cloud[idx] += noise
        
        return result_cloud

    """
    taken from https://github.com/thu-ml/3D_Corruptions_AD (last access: 11.01.2026)
    """
    def impulse_noise(self, cloud, severity):
        """
        Simulates Impulse Noise (Salt-and-Pepper): Sudden extreme spikes.
        This replaces standard reading errors with distinct, sharp offsets.

        Instead of shifting a point slightly (like Gaussian noise), 
        this adds a distinct, sharp value (e.g., +0.1 or -0.1) to random points. 
        It creates a "grainy" look with sharp artifacts.
        """
        if cloud is None or len(cloud) == 0:
            return cloud

        N, C = cloud.shape
        # Severity determines number of affected points (N/30 down to N/10)
        divisors = [30, 25, 20, 15, 10]
        sev_idx = np.clip(severity - 1, 0, 4)
        c = N // divisors[sev_idx]

        if c == 0:
            return cloud

        idx = self.rng.choice(N, c, replace=False)
        
        # Simulates spikes by adding discrete values (-1 or 1) * scale
        # Note: The scale 0.1 is from the reference, but "extreme" impulse 
        # often implies larger spikes in real-world scenarios.
        spikes = self.rng.choice([-1, 1], size=(c, C)) * 0.1
        
        result_cloud = cloud.copy()
        result_cloud[idx] += spikes
        
        return result_cloud

    # =============================================================================
    # DROPOUT METHODS
    # =============================================================================

    """
    The functions random_dropout(), structured_dropout(), reduce_fov(), 
    add_gaussian_noise(), simulate_motion_distortion(),
    simulate_occlusion(), sparsce_scan() 
    are all adapted from:
    "https://github.com/mawuto/lidar_augmentation_cpp_ws" (last access: 11.01.2026)
    """

    def random_dropout(self, cloud, rate):
        """
        Randomly removes points from the cloud.
        :param cloud: (N, C) numpy array (xyz...)
        :param rate: Float [0, 1], percentage of points to drop.
        :return: (M, C) numpy array, boolean mask
        """
        if cloud is None or len(cloud) == 0:
            logger.warning("Random dropout: input cloud is empty")
            return cloud, np.array([], dtype=bool)

        n_points = cloud.shape[0]
        keep_rate = 1.0 - np.clip(rate, 0.0, 1.0)

        if keep_rate <= 0.0:
            return np.empty((0, cloud.shape[1])), np.zeros(n_points, dtype=bool)

        mask = self.rng.random(n_points) < keep_rate
        result_cloud = cloud[mask]

        logger.debug(f"Random dropout: {n_points} -> {len(result_cloud)} points")
        return result_cloud, mask

    def structured_dropout(self, cloud, pattern, rate, ring_col_idx=None):
        """
        Drops points based on geometric patterns.
        :param cloud: (N, C) numpy array.
        :param pattern: 'ring', 'sector', 'distance', 'checkerboard'
        :param ring_col_idx: Index of the column containing ring/channel info.

        Ring/Line: Simulates a specific laser diode failing (e.g., in a 64-beam LiDAR, beam #32 dies).
        Sector: Simulates a blockage on the sensor cover (e.g., a mud splatter blocking 45 degrees of the view).
        Distance: Simulates failure to detect objects at specific ranges.
        Checkerboard: A synthetic pattern used to test robustness against periodic data loss.
        """
        
        if cloud is None or len(cloud) == 0:
            return cloud, np.array([], dtype=bool)

        n_points = cloud.shape[0]
        mask = np.ones(n_points, dtype=bool)
        rate = np.clip(rate, 0.0, 1.0)

        if rate <= 0.0:
            return cloud, mask

        # Not implemented yet
        # if pattern in ["ring", "line"]:
            # mask = self._create_ring_mask(cloud, rate, ring_col_idx)
        if pattern == "sector":
            mask = self._create_sector_mask(cloud, rate)
        elif pattern == "distance":
            mask = self._create_distance_mask(cloud, rate)
        elif pattern == "checkerboard":
            mask = self._create_checkerboard_mask(cloud, rate)
        else:
            logger.warning(f"Unknown pattern {pattern}, using random")
            return self.random_dropout(cloud, rate)

        result_cloud = cloud[mask]
        logger.debug(f"Structured dropout ({pattern}): {n_points} -> {len(result_cloud)}")
        return result_cloud, mask

    # =============================================================================
    # FOV REDUCTION
    # =============================================================================

    def reduce_fov(self, cloud, fov_reduction):
        """
        Cuts points outside specified angles.
        :param fov_reduction: Dict {'horizontal': float, 'vertical': float}
        """
        if cloud is None or len(cloud) == 0:
            return cloud

        h_red = np.clip(fov_reduction.get('horizontal', 0.0), 0.0, 0.9)
        v_red = np.clip(fov_reduction.get('vertical', 0.0), 0.0, 0.9)

        if h_red <= 0 and v_red <= 0:
            return cloud.copy()

        x, y, z = cloud[:, 0], cloud[:, 1], cloud[:, 2]

        mask = np.ones(len(cloud), dtype=bool)

        if h_red > 0:
            h_fov = 2 * np.pi * (1.0 - h_red)
            h_limit = h_fov / 2.0
            azimuth = np.arctan2(y, x)
            mask &= (np.abs(azimuth) <= h_limit)

        if v_red > 0:
            v_fov = np.pi * (1.0 - v_red)
            v_limit = v_fov / 2.0
            r = np.linalg.norm(cloud[:, :3], axis=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                elevation = np.arcsin(z / r)
                elevation = np.nan_to_num(elevation)
            mask &= (np.abs(elevation) <= v_limit)

        return cloud[mask]

    # =============================================================================
    # NOISE INJECTION
    # =============================================================================

    def add_gaussian_noise(self, cloud, noise_params:Dict):
        """
        Adds Gaussian noise to coordinates.
        """
        if cloud is None or len(cloud) == 0:
            return cloud

        result_cloud = cloud.copy()
        
        g_std = max(0.0, noise_params.get('gaussian_std', 0.0))
        o_rate = np.clip(noise_params.get('outlier_rate', 0.0), 0.0, 1.0)
        o_std = max(0.0, noise_params.get('outlier_std', 0.0))

        if g_std > 0.0:
            noise = self.rng.normal(0, g_std, (len(cloud), 3))
            result_cloud[:, :3] += noise

        if o_rate > 0.0 and o_std > 0.0:
            n_outliers = int(len(cloud) * o_rate)
            indices = self.rng.choice(len(cloud), n_outliers, replace=False)
            outlier_noise = self.rng.normal(0, o_std, (n_outliers, 3))
            result_cloud[indices, :3] += outlier_noise

        return result_cloud

    # =============================================================================
    # MOTION DISTORTION
    # =============================================================================

    def simulate_motion_distortion(self, cloud, timestamps, motion_params):
        """
        Warps points based on velocity and time.
        """
        if cloud is None or len(cloud) == 0:
            return cloud
        
        if len(timestamps) != len(cloud):
            logger.warning("Timestamp length mismatch")
            return cloud

        result_cloud = cloud.copy()
        
        lin_vel = np.array(motion_params.get('linear_velocity', [0,0,0]))
        ang_vel = np.array(motion_params.get('angular_velocity', [0,0,0]))

        if np.linalg.norm(lin_vel) < 1e-6 and np.linalg.norm(ang_vel) < 1e-6:
            return result_cloud

        t_min = np.min(timestamps)
        t_max = np.max(timestamps)
        
        if t_min == t_max:
            return result_cloud
            
        time_span = float(t_max - t_min)
        if time_span > 1e9: 
            scale = 1e-9
        else:
            scale = 1.0
            
        rel_time = (timestamps - t_min) * scale

        linear_displacement = np.outer(rel_time, lin_vel) 
        rot_vectors = np.outer(rel_time, ang_vel)
        rotations = R.from_rotvec(rot_vectors)
        
        original_pos = result_cloud[:, :3]
        rotated_pos = rotations.apply(original_pos)
        
        result_cloud[:, :3] = rotated_pos + linear_displacement
        
        return result_cloud

    # =============================================================================
    # OCCLUSION SIMULATION
    # =============================================================================

    def simulate_occlusion(self, cloud, occlusion_params):
        """
        Removes spherical patches of points.
        Real-world scenario: Simulates an object (like a bird, a drone, or a splash of mud) blocking a chunk of the view.
        """
        if cloud is None or len(cloud) == 0:
            return cloud

        dist_thresh = max(0.1, occlusion_params.get('distance_threshold', 50.0))
        patch_count = max(1, int(occlusion_params.get('random_patches_count', 3)))
        patch_size = max(0.1, occlusion_params.get('random_patches_size', 1.5))

        dists = np.linalg.norm(cloud[:, :3], axis=1)
        valid_indices = np.where(dists <= dist_thresh)[0]
        
        if len(valid_indices) == 0:
            return cloud

        mask = np.ones(len(cloud), dtype=bool)

        for _ in range(patch_count):
            center_idx = self.rng.choice(valid_indices)
            center_point = cloud[center_idx, :3]
            d_to_center = np.linalg.norm(cloud[:, :3] - center_point, axis=1)
            mask &= (d_to_center > patch_size)

        return cloud[mask]

    # =============================================================================
    # SPARSE SCAN
    # =============================================================================

    def sparse_scan_pattern(self, cloud, sparsity_factor):
        """
        Downsamples the cloud by taking every kth point.
        """
        if cloud is None or len(cloud) == 0:
            return cloud
            
        factor = max(1, int(sparsity_factor))
        if factor == 1:
            return cloud.copy()
            
        return cloud[::factor]

    # =============================================================================
    # HELPERS
    # =============================================================================

    """
    Helper functions creating the masks for the structured point cloud corruptions.
    """
    def _create_ring_mask(self, cloud, rate, ring_idx):
        n_points = cloud.shape[0]
        if ring_idx is None or ring_idx >= cloud.shape[1]:
            logger.warning("No ring information available")
            return np.ones(n_points, dtype=bool)

        rings = cloud[:, ring_idx].astype(int)
        unique_rings = np.unique(rings)
        
        n_drop = int(len(unique_rings) * rate)
        rings_to_drop = self.rng.choice(unique_rings, n_drop, replace=False)
        return ~np.isin(rings, rings_to_drop)


    def _create_sector_mask(self, cloud, rate):
        n_sectors = 8
        sector_angle = 2.0 * np.pi / n_sectors
        
        sectors_to_drop = self.rng.choice(n_sectors, int(n_sectors * rate), replace=False)
        
        azimuth = np.arctan2(cloud[:, 1], cloud[:, 0])
        azimuth = np.where(azimuth < 0, azimuth + 2 * np.pi, azimuth)
        
        point_sectors = (azimuth / sector_angle).astype(int)
        point_sectors = np.clip(point_sectors, 0, n_sectors - 1)
        
        return ~np.isin(point_sectors, sectors_to_drop)


    def _create_distance_mask(self, cloud, rate):
        dists = np.linalg.norm(cloud[:, :3], axis=1)
        min_d, max_d = np.min(dists), np.max(dists)
        
        n_bands = 10
        band_size = (max_d - min_d) / n_bands
        bands_to_drop = self.rng.choice(n_bands, int(n_bands * rate), replace=False)
        
        point_bands = ((dists - min_d) / band_size).astype(int)
        point_bands = np.clip(point_bands, 0, n_bands - 1)
        
        return ~np.isin(point_bands, bands_to_drop)


    def _create_checkerboard_mask(self, cloud, rate):
        az_divs = 16
        el_divs = 8
        
        azimuth = np.arctan2(cloud[:, 1], cloud[:, 0])
        azimuth = np.where(azimuth < 0, azimuth + 2 * np.pi, azimuth)
        
        r = np.linalg.norm(cloud[:, :3], axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            elevation = np.arcsin(cloud[:, 2] / r)
            elevation = np.nan_to_num(elevation)
        elevation += np.pi / 2.0
        
        az_step = 2.0 * np.pi / az_divs
        el_step = np.pi / el_divs
        
        az_idx = np.clip((azimuth / az_step).astype(int), 0, az_divs - 1)
        el_idx = np.clip((elevation / el_step).astype(int), 0, el_divs - 1)
        
        square_ids = el_idx * az_divs + az_idx
        unique_ids = np.unique(square_ids)
        
        candidate_squares = []
        for uid in unique_ids:
            u_az = uid % az_divs
            u_el = uid // az_divs
            if (u_az + u_el) % 2 == 0:
                candidate_squares.append(uid)
                
        n_drop = int(len(unique_ids) * rate)
        
        if len(candidate_squares) > 0:
            squares_to_drop = self.rng.choice(candidate_squares, min(n_drop, len(candidate_squares)), replace=False)
            return ~np.isin(square_ids, squares_to_drop)
        
        return np.ones(len(cloud), dtype=bool)
