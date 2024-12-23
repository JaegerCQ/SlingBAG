**Ultrafast Simulation Method for 3D Photoacoustic Imaging**  
This simulation method completes in mere seconds what k-Wave requires tens of hours to compute, while reducing memory consumption by orders of magnitude. The simulation results achieve a similarity of over 99.9% with those generated using k-Wave.

As a by-product of our research on SlingBAG, we developed a novel approach: utilizing the **forward process of SlingBAG** as a fast method to generate simulated photoacoustic data. In this approach, users can generate simulation results rapidly by using the provided **`SlingBAG_forward_simulation.ipynb`** script. This script takes two primary input files:
1. **`data/sensor_location_for_simulation.txt`**: The sensor coordinate information for the detector array.
2. **`data/phantom_for_simulation.ply`**: The simulated photoacoustic signal source information.  

Additionally, users need to provide the following parameters (we haved provided the default value in the code):
- **`num_times`**: Number of signal sampling points for a single detector,  
- **`dt`**: Time interval between sampling points (default: 25e-9, i.e., the inverse of the DAQ sampling frequency, 40MHz),  
- **`Vs`**: The speed of sound (default: 1500 m/s).  

The generated simulation data will be saved in **`simulated_sensor_signal.txt`**. The library functions and environmental configuration required for running this code are identical to the ones used for the SlingBAG iterative reconstruction algorithm.

---

**Example Data Input Description:**  
1. **`data/sensor_location_for_simulation.txt`**:   
This file is a (1013, 3) two-dimensional array, representing 1013 detectors in the array. Each row contains the three-dimensional coordinates of a detector (unit: meters).  

![image](https://github.com/JaegerCQ/SlingBAG/blob/main/figures/location_show.png)  

2. **`data/phantom_for_simulation.ply`**:  
This file contains 21,744 elements, representing the photoacoustic point sources in the simulated scene. Each element has 5 attributes:
   - The **first three attributes** are the three-dimensional coordinates of the point source (in meters).  
   - The **fourth attribute** is the photoacoustic pressure.  
   - The **fifth attribute** is the resolution, which corresponds to the point spread function (PSF). This is analogous to the grid resolution setting in k-Wave's simulation process (unit: meters).  
![image](https://github.com/JaegerCQ/SlingBAG/blob/main/figures/ply_show.png)  
---

The detector and point source coordinates are both defined in the global/world coordinate system. This ultrafast simulation method may provide a convenient and efficient approach for generating high-quality photoacoustic simulation data, if you have any questions while using it, please be free to tell us, thanks for your feedback!
