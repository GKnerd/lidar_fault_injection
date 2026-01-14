"""
GLARE
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

# CONFIGURATION
INPUT_DIR = "/home/georgios-katranis/Projects/CogniSafe3D/datasets/lihra/collaboration/lidar/pointclouds"
OUTPUT_DIR = "./lidar_data/exp_01_glare"

injector = LidarInjector(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR)

# Process 20 files to test
injector.set_sampling(mode="count", value=20, seed=42)
injector.set_workers(4)

# 100% of data gets Glare
injector.add_group("Direct Sunlight", fraction=1.0) \
    .apply("Sunlight Glare", severity=4)

print(f"Running Experiment 01: Sunlight Glare -> {OUTPUT_DIR}")
injector.run()
