**Ultrafast Simulation Method for 3D Photoacoustic Imaging**  
This simulation method completes in mere seconds what k-Wave requires tens of hours to compute, while reducing memory consumption by orders of magnitude. The simulation results achieve a similarity of over 99.9% with those generated using k-Wave.

As a by-product of our research on SlingBAG, we developed a novel approach: utilizing the **forward process of SlingBAG** as a fast method to generate simulated photoacoustic data. In this approach, users can generate simulation results rapidly by using the provided **`SlingBAG_forward_simulation.ipynb`** script. This script takes two primary input files:
1. **`data/sensor_location_for_simulation.txt`**: The sensor coordinate information for the detector array.
2. **`data/phantom_for_simulation.ply`**: The simulated photoacoustic signal source information.  

Additionally, users need to provide the following parameters (we haved provided the default value in the code):
- **`num_times`**: Number of signal sampling points for a single detector (default: 4096),  
- **`dt`**: Time interval between sampling points (unit: seconds, default: 25e-9, i.e., the inverse of the DAQ sampling frequency, 40MHz),  
- **`Vs`**: The speed of sound (default: 1500 m/s).  

The generated simulation data will be saved in **`simulated_sensor_signal.txt`**. The library functions and environmental configuration required for running this code are identical to the ones used for the SlingBAG iterative reconstruction algorithm.

We also present a demo that directly converts the 3D simulation photoacoustic source stored in a .mat file into the required .ply file format. The original .mat file is provided at `data/source.mat`, and the conversion code can be found in the main directory as `mat2ply.m`. This code reads the 3D volume, centers it in the world coordinate system, and converts all points with values greater than 0 into point cloud data for storage. Additionally, the code calculates the coordinate values based on the physical spacing represented by the user-defined unit grid (default physical spacing of the unit grid: 0.1mm). The final result is saved in the file `data/phantom_for_simulation.ply`.

---

**Example Data Input and Simulation output Description:**  
1. **`data/sensor_location_for_simulation.txt`**:   
This file is a (1013, 3) two-dimensional array, representing 1013 detectors in the array. Each row contains the three-dimensional coordinates of a detector (unit: meters).  

![image](https://github.com/JaegerCQ/SlingBAG/blob/main/figures/location_show.png)  

2. **`data/phantom_for_simulation.ply`**:  
This file contains 21,744 elements, representing the photoacoustic point sources in the simulated scene. Each element has 5 attributes:
   - The **first three attributes** are the three-dimensional coordinates of the point source (in meters).  
   - The **fourth attribute** is the photoacoustic pressure.  
   - The **fifth attribute** is the resolution, which corresponds to the point spread function (PSF). This is analogous to the grid resolution setting in k-Wave's simulation process (unit: meters).  
![image](https://github.com/JaegerCQ/SlingBAG/blob/main/figures/ply_show.png)

3. **`simulated_sensor_signal.txt`**:  
This file is the output result of the simulation. In the given example, the output result is a 2D array of size (1013, 4096), where 1013 represents the data from 1013 probe channels, and each channel contains 4096 sampling points. The output data corresponds one-to-one with the probe coordinates. Users can modify the saved output format as needed, such as saving it in `.txt`, `.dat`, or `.mat` file formats.
---

The detector and point source coordinates are both defined in the global/world coordinate system. This ultrafast simulation method may provide a convenient and efficient approach for generating high-quality photoacoustic simulation data, if you have any questions while using it, please be free to tell us, thanks for your feedback!
