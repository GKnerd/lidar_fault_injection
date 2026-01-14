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
