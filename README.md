# Sliding Gaussian ball adaptive growth: point cloud-based iterative algorithm for large-scale 3D photoacoustic imaging
Large-scale photoacoustic (PA) 3D imaging has become increasingly important for both clinical and pre-clinical applications. Limited by resource and application constrains, only sparsely-distributed transducer arrays can be applied, which necessitates advanced image reconstruction algorithms to overcome artifacts caused by using back-projection algorithm. However, high computing memory consumption of traditional iterative algorithms for large-scale 3D cases is practically unacceptable. Here, we propose a point cloud-based iterative algorithm that reduces memory consumption by several orders, wherein a 3D photoacoustic scene is modeled as a series of Gaussian-distributed spherical sources. During the iterative reconstruction process, the properties of each Gaussian source, including peak intensities, standard deviations and means are stored in form of point cloud, then continuously optimized and adaptively undergoing destroying, splitting, and duplication along the gradient direction, thus manifesting the sliding ball adaptive growth effect. This method, named the sliding Gaussian ball adaptive growth (SlingBAG) algorithm, enables high-quality 3D large-scale PA reconstruction with fast iteration and extremely less memory usage. We validated SlingBAG algorithm in both simulation study and in vivo animal experiments.