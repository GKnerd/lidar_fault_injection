"""
DROPOUT
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
OUTPUT_DIR = "exp_05_dropout"

injector = LidarInjector(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR)
injector.set_sampling(mode="count", value=20, seed=42)
injector.set_workers(4)

# 100% of data gets Sector Dropout
# Simulates mud blocking a specific angle of the sensor
injector.add_group("Mud Splatter", fraction=1.0) \
    .apply("Structured Dropout", pattern="sector", rate=0.25)

print(f"Running Experiment 07: Structured Dropout (Sector) -> {OUTPUT_DIR}")
injector.run()
