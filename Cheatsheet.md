# Parameter Reference

| Corruption Name | Parameter | Type | Description |
| :--- | :--- | :--- | :--- |
| **Sunlight Glare** | `severity` | `int` | Intensity level (1-5). |
| **Crosstalk** | `severity` | `int` | Intensity level (1-5). |
| **Impulse Noise** | `severity` | `int` | Intensity level (1-5). |
| **Random Dropout** | `rate` | `float` | Percentage of points to drop (0.0 - 1.0). |
| **Structured Dropout** | `pattern` | `str` | `'sector'`, `'distance'`, or `'checkerboard'`. |
| | `rate` | `float` | Intensity of dropout (0.0 - 1.0). |
| **Motion Distortion** | `linear_velocity` | `list` | `[x, y, z]` velocity in m/s (e.g., `[20.0, 0, 0]`). |
| | `angular_velocity` | `list` | `[roll, pitch, yaw]` in rad/s (e.g., `[0, 0, 0.5]`). |
| **Occlusion** | `distance_threshold` | `float` | Max distance to spawn occlusions (meters). |
| | `random_patches_count` | `int` | Number of occlusion spheres. |
| | `random_patches_size` | `float` | Radius of occlusion spheres (meters). |
| **Gaussian Noise** | `gaussian_std` | `float` | Standard deviation of position jitter (meters). |
| | `outlier_rate` | `float` | Fraction of points to turn into outliers. |
| | `outlier_std` | `float` | Sigma for outlier noise. |
| **FOV Reduction** | `horizontal` | `float` | Percentage to crop from azimuth (0.0 - 1.0). |
| | `vertical` | `float` | Percentage to crop from elevation (0.0 - 1.0). |
| **Sparse Scan** | `sparsity_factor` | `int` | Skip every Nth point (e.g., `2` halves resolution). |
