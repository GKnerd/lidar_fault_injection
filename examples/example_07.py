"""
MULTIPLE DIFFERENT CORRUPTIONS
"""
# Imports for the code to work inside the examples directory
# -----------------------------------------------------------
import sys
import os

# Add the parent directory (root) to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# -----------------------------------------------------------

from src.lidar_fault_injector import LidarInjector

INPUT_DIR = "/home/georgios-katranis/Projects/CogniSafe3D/datasets/lihra/collaboration/lidar/pointclouds"
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../lidar_data/exp_ratio_complex"))

injector = LidarInjector(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR)

# 2. SAMPLING (Ratio Mode)
# "value=0.5" means we randomly select 50% of ALL files in the input directory.
# The groups below will split up THIS 50% subset.
injector.set_sampling(mode="ratio", value=0.5, seed=777)
injector.set_workers(4)

# 3. DEFINE GROUPS (Must sum to <= 1.0)
# Note: 'fraction' here is relative to the sampled set (the 50%), not the total dataset.

# --- Group A: Heavy Structural Damage (60% of the sample) ---
# Simulates a vehicle with a broken sensor driving fast.
# - Motion Distortion (Fast driving)
# - Structured Dropout (Dead beams)
# - Occlusion (Debris on lens)
injector.add_group("Heavy Damage", fraction=0.6) \
    .apply("Motion Distortion", 
           linear_velocity=[35.0, 0, 0], 
           angular_velocity=[0, 0, 0.1]) \
    .apply("Structured Dropout", 
           pattern="checkerboard", 
           rate=0.2) \
    .apply("Occlusion", 
           distance_threshold=10.0, 
           random_patches_count=5)

# --- Group B: Light Environmental Noise (40% of the sample) ---
# Simulates a standard sunny day with some sensor noise.
# - Sunlight Glare (Blinding light)
# - Gaussian Noise (Standard jitter)
injector.add_group("Light Env Damage", fraction=0.4) \
    .apply("Sunlight Glare", severity=2) \
    .apply("Gaussian Noise", gaussian_std=0.02)

# 4. RUN
print(f"Running Ratio Experiment: Processing 50% of dataset -> {OUTPUT_DIR}")
injector.run()
