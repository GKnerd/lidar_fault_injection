"""
MULTIPLE NOISES ON THE ENTIRE DATASET
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
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../lidar_data/exp_complex_storm")

# Initialize
injector = LidarInjector(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR)

# Process 50 files to verify the compound effect
injector.set_sampling(mode="count", value=50, seed=999)
injector.set_workers(4)

# We apply a chain of 5 distinct corruptions to 100% of the sampled data.

injector.add_group("Stormy Highway Failure", fraction=1.0) \
    .apply("Motion Distortion", 
           linear_velocity=[28.0, 0, 0],   
           angular_velocity=[0, 0, 0.05]) \
    .apply("Gaussian Noise", 
           gaussian_std=0.08,              
           outlier_rate=0.03,              
           outlier_std=0.5)  \
    .apply("Random Dropout", 
           rate=0.10)   \
    .apply("Structured Dropout", 
           pattern="sector",               
           rate=0.15)   \
    .apply("Crosstalk", 
           severity=2)                     

# RUN
print(f"Starting Complex Simulation: Stormy Highway -> {OUTPUT_DIR}")
injector.run()
