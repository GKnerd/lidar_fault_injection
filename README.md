# LiDAR Fault Injector


The `LidarAugmenter` class currently supports 10 distinct corruption types of `geometric` and `sensor-specific corruptions` designed to 
simulate real-world failure cases. 

### Supported Corruptions

| Corruption | Description |
| :--- | :--- |
| **Sunlight Glare** | Simulates bright light sources (e.g., direct sun) causing blinding noise. |
| **Crosstalk** | Simulates interference between laser emitters or returns from different beams. |
| **Impulse Noise** | Simulates "salt-and-pepper" digital transmission errors. |
| **Random Dropout** | Stochastically removes a percentage of points across the entire cloud. |
| **Structured Dropout** | Simulates patterns like **Sector** (mud blockage), **Distance** (range bands), or **Checkerboard**. |
| **Motion Distortion** | Simulates skewing effects caused by vehicle motion during mechanical spins. |
| **Occlusion** | Simulates physical objects blocking the near-field view. |
| **Gaussian Noise** | Adds standard measurement jitter and random atmospheric outliers (dust/bugs). |
| **FOV Reduction** | Artificially limits horizontal (azimuth) or vertical (elevation) angles. |
| **Sparse Scan** | Downsamples resolution by skipping points to simulate lower-end sensors. |

--- 

## Quick Start

### 1. Installation
Clone the repository and set up the environment:

```bash
git clone git@github.com:GKnerd/lidar_fault_injection.git
cd lidar_fault_injection

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate  

# Install dependencies
pip install -r requirements.txt
```
### 2. Run an Experiment
In the root of your directory you can then create a experiment_xx.py file and use the Python Builder to define
your experiment on the LiDAR data. 

```python
from src.lidar_injector import LidarInjector

# Initialize
injector = LidarInjector(
    input_dir="./data/raw_pcds", 
    output_dir="./data/augmented"
)

# Configure Sampling (e.g., process 50 files)
injector.set_sampling(mode="count", value=50)

# Define a scenario: 100% of data gets Glare + Noise
injector.add_group("Sunny Day", fraction=1.0) \
    .apply("Sunlight Glare", severity=3) \
    .apply("Gaussian Noise", gaussian_std=0.02)

# Run
injector.run()
```
### 3. Output 

The tool will automatically create a timestamped output directory (e.g., augmented_2026-01-14_14-30-00) containing:
- The augmented .pcd files.
- A metadata.json file logging exactly which corruptions were applied to each file.


### 4. Additional Information
- Check the examples/ folder for complex scenarios and refer to the Parameter Cheat Sheet below to see valid ranges for every corruption type.
- If you want to save your experiments inside a directory to keep the top-level of the root folder free from clutter. Then add this your .py files

```
 -----------------------------------------------------------
import sys
import os

# Add the parent directory (root) to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# -----------------------------------------------------------
```
