from src.lidar_fault_injector import LidarInjector

"""
LiDAR Fault Injection Experiment Runner
========================================
This script configures and executes a data augmentation pipeline using the LidarInjector.
It demonstrates the "Builder Pattern" to create diverse subsets of corrupted data.

Usage (from the root of your source directory):
    python3 run_experiment.py
"""


# ==============================================================================
# 1. INITIALIZATION
# ==============================================================================
# Define where your raw point clouds (.pcd) are located and where to save the results.
# The output directory will be created automatically if it doesn't exist.

injector = LidarInjector(
    input_dir   =   "/home/georgios-katranis/Projects/CogniSafe3D/datasets/lihra/collaboration/lidar/pointclouds",
    output_dir  =   "./lidar_data/augmented"
)

# ==============================================================================
# 2. SAMPLING CONFIGURATION
# ==============================================================================
# Decide how many files to process from the input directory.
# Options for 'mode':
#   - "count": Process a fixed number of files (e.g., value=50).
#   - "ratio": Process a percentage of the total files (e.g., value=0.1 for 10%).

injector.set_sampling(mode="count", value=50, seed=42)

# Set the number of parallel CPU processes.
# Recommended: Set this to the number of CPU cores available on your machine.
injector.set_workers(4)

# ==============================================================================
# 3. DEFINE AUGMENTATION SCENARIOS (GROUPS)
# ==============================================================================
# Instead of applying the same corruption to every file, we split the sampled data
# into "Groups". The 'fraction' arguments across all groups should ideally sum to 1.0 (100%),
# though the tool handles cases where they don't by just processing what is requested.

# --- GROUP A: Sunny Day Scenario (30% of data) ---
# Simulates bright sunlight causing sensor blinding (Glare) and standard atmospheric scattering (Noise).
# Chain .apply() calls to stack multiple corruptions on the same file.
injector.add_group("Sunny Day", fraction=0.3) \
    .apply("Sunlight Glare", severity=3) \
    .apply("Gaussian Noise", gaussian_std=0.01)

# --- GROUP B: Broken Sensor Scenario (20% of data) ---
# Simulates hardware failures.
# 'Structured Dropout' removes specific rings (mimicking a dead laser diode).
# 'Crosstalk' adds interference spikes near valid points.
injector.add_group("Hardware Failure", fraction=0.2) \
    .apply("Structured Dropout", pattern="ring", rate=0.15) \
    .apply("Crosstalk", severity=4)

# --- GROUP C: High Speed Scenario (50% of data) ---
# Simulates a vehicle driving at high speeds.
# 'Motion Distortion' skews the cloud based on velocity vectors [x, y, z].
# 'FOV Reduction' simulates a limited view (e.g., hood occlusion or region-of-interest cropping).
injector.add_group("High Speed", fraction=0.5) \
    .apply("Motion Distortion", linear_velocity=[25.0, 0, 0], angular_velocity=[0, 0, 0.2]) \
    .apply("Field of View (FOV) Reduction", horizontal=0.1)

# ==============================================================================
# 4. EXECUTION
# ==============================================================================
# Compiles the configuration, loads files lazily, processes them in parallel, 
# and saves a 'metadata.json' file in the output directory detailing every change.
print("Starting Augmentation Experiment...")
injector.run()
