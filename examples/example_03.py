"""
IMPULSE NOISE
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
OUTPUT_DIR = "./lidar_data/exp_03_impulse"

injector = LidarInjector(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR)
injector.set_sampling(mode="count", value=20, seed=42)
injector.set_workers(4)

# 100% of data gets Impulse Noise
injector.add_group("Digital Errors", fraction=1.0) \
    .apply("Impulse Noise", severity=5)

print(f"Running Experiment 03: Impulse Noise -> {OUTPUT_DIR}")
injector.run()
