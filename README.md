# Sliding Gaussian ball adaptive growth: point cloud-based iterative algorithm for large-scale 3D photoacoustic imaging

[***`ArXiv`***](https://arxiv.org/abs/2407.11781)

Large-scale photoacoustic (PA) 3D imaging has become increasingly important for both clinical and pre-clinical applications. Limited by resource and application constrains, only sparsely-distributed transducer arrays can be applied, which necessitates advanced image reconstruction algorithms to overcome artifacts caused by using back-projection algorithm. However, high computing memory consumption of traditional iterative algorithms for large-scale 3D cases is practically unacceptable. Here, we propose a point cloud-based iterative algorithm that reduces memory consumption by several orders, wherein a 3D photoacoustic scene is modeled as a series of Gaussian-distributed spherical sources. During the iterative reconstruction process, the properties of each Gaussian source, including peak intensities, standard deviations and means are stored in form of point cloud, then continuously optimized and adaptively undergoing destroying, splitting, and duplication along the gradient direction, thus manifesting the sliding ball adaptive growth effect. This method, named the sliding Gaussian ball adaptive growth (SlingBAG) algorithm, enables high-quality 3D large-scale PA reconstruction with fast iteration and extremely less memory usage. We validated SlingBAG algorithm in both simulation study and in vivo animal experiments.  

![image](https://github.com/JaegerCQ/SlingBAG/blob/main/figures/pipeline_gaussian.png)   
_The overview framework of Sliding Gaussian Ball Adaptive Growth algorithm. (a) The SlingBAG pipeline. (b) The forward simulation of differentiable rapid radiator. (c) Adaptive growth optimization based on iterative point cloud._

## Display of SlingBAG's results

![image](https://github.com/JaegerCQ/SlingBAG/blob/main/figures/rat_liver_recon.gif) 
_Comparison of reconstruction results of rat liver between UBP and SlingBAG._    


![image](https://github.com/JaegerCQ/SlingBAG/blob/main/figures/hand_vessel_recon.gif) 
_Comparison of reconstruction results of hand vessels between UBP and SlingBAG using sparse 576 sensors._    

## Guidance

The example in the provided codes is for the reconstruction of simulated hand vessel with 196 elements planar array (detailed in the article), if you want to reconstrut your own data, please replace the sensor location and sensor data files in the `train_196_elements_coarse_recon.ipynb` and `train_196_elements_fine_recon.ipynb`. Besides, the boundary set of the Gaussian balls should be modified carefully to match the reconstruction area. Sorry for all the inconvenience, we promise that the SlingBAG will soon be much more user-friendly, and we hope this guidance may help you. Good luck my friends!  
If you have any questions while using SlingBAG for 3D reconstruction, please be free to contact us. Best wishes!

## BibTeX

```
@article{li2024slingbag,
  title={Sliding Gaussian ball adaptive growth (SlingBAG): point cloud-based iterative algorithm for large-scale 3D photoacoustic imaging},
  author={Li, Shuang and Wang, Yibing and Gao, Jian and Kim, Chulhong and Choi, Seongwook and Zhang, Yu and Chen, Qian and Yao, Yao and Li, Changhui},
  journal={arXiv preprint arXiv:2407.11781},
  year={2024}
}
```

## Installation

```bash
git clone https://github.com/JaegerCQ/SlingBAG.git
cd SlingBAG
```

```bash
conda create -n SlingBAG python=3.10 -y
conda activate SlingBAG
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

```bash
pip install -r requirements.txt
```

Warning: for Windows, it needs `setuptools <= 72.1.0`; for Linux, it needs `gcc >= 9.1`.

## Usage

### Coarse reconstruction
Run `train_196_elements_coarse_recon.ipynb`.

### Fine reconstruction
Run `train_196_elements_fine_recon.ipynb`.

### Conversion from point cloud to voxel grid
Run `point_cloud_to_voxel_grid_shader.ipynb`.
