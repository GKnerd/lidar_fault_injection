# LiDAR Fault Injector


The `LidarAugmenter` class currently supports a wide range of `geometric` and `sensor-specific corruptions` designed to 
simulate real-world failure cases. 

The supported corruptions are:

### Sunlight Glare
Simulates the sensor facing a bright light source (e.g., direct sunlight). 

### Crosstalk 
Simulates interference between laser emitters or returns from different beams.

### Impulse Noise 
Simulates "salt-and-pepper" noise or digital transmission errors.

### Random Dropout 
Stochastically removes a percentage of points across the entire cloud.

### Structured Dropout 
- Sector: Simulates a physical blockage on the sensor cover (e.g., mud splatter blocking a 45° angle).
- Distance: Simulates failure to detect objects at specific range bands.
- Checkerboard: A synthetic pattern used to test model robustness against periodic data loss.

### Motion Distortion 
- Simulates the "shearing" or skewing effect caused by a vehicle moving while a mechanical LiDAR spins.

### Occlusion
Simulates physical objects blocking the view close to the sensor.

### Gaussian Noise 
Adds standard measurement error (jitter) to point coordinates. Can also add random "outlier" points to simulate atmospheric scattering (dust/bugs).

### Field of View (FOV) Reduction 
Artificially limits the sensor's horizontal (azimuth) or vertical (elevation) angles.

### Sparse Scan 
Downsamples the point cloud resolution by systematically skipping points, simulating a lower-resolution sensor.
