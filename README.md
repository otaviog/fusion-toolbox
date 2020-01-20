# Fusion Toolbox

A toolkit for prototyping RGBD-based simultaneous localization and
mapping (SLAM) algorithms with PyTorch.

The aim of this project is to faciliciate load, process, visualization
and evaluation of 3D reconstruction pipelines. We also offer
registration algorithm and fusion of surfels using index map.

This work is sponsered by Eldorado Research Institute.

## Features

### Dataset parsing:

```python
from fiontb.dataset.tumrgbd import load_tumrgbd

dataset = load_tumrgbd("scene1")

frame0 = dataset[0]
print(frame)
print(frame.depth_image.shape)
print(frame.rgb_image.shape)
```

>Frame: Frame with shape (480x640) depth: True, RGB: True, segmentation: False, Normal: False
>
>Frame's description: Depth intrinsics: {Intrinsic matrix: tensor([[525.0000,   0.0000, 319.5000],
>        [  0.0000, 525.0000, 239.5000],
>        [  0.0000,   0.0000,   1.0000]]), radial distortion coefficients: [], image size: (640, 480)}, depth scale: 0.0002, depth bias: 0.0, timestamp: 1305031453.374112, Rigid Transformation: {{'matrix': tensor([[ 0.8758,  0.3425, -0.3400,  1.3137],
>        [ 0.4820, -0.5867,  0.6508,  0.8486],
>        [ 0.0234, -0.7338, -0.6789,  1.5192],
>        [ 0.0000,  0.0000,  0.0000,  1.0000]], dtype=torch.float64)}}

### Visualization:

```python
from fiontb.data.ilrgbd import load_ilrgbd
dataset = load_ilrgbd("apartment", "apartment/apartment.log")

from fiontb.viz.datasetviewer import DatasetViewer
DatasetViewer(dataset).run()
```

![](doc/samples/viz.png)

### Processing:

```python
from fiontb.data.ftb import load_ftb
dataset = load_ftb("test-data/rgbd/sample2/")
frame = dataset[0]

from fiontb.processing import bilateral_depth_filter
frame.depth_image = bilateral_depth_filter(
    frame.depth_image, frame.depth_image > 0,
    filter_width=6, sigma_color=30, sigma_space=5)
```

### Data structures:

```python
from fiontb.data.ftb import load_ftb

dataset = load_ftb("test-data/rgbd/sample2")
frame = dataset[0]

##
# A point cloud with per-point attributes arraged as its source image.

from fiontb.frame import FramePointCloud
fpcl = FramePointCloud.from_frame(frame)

print("FramePointCloud:")
print("`.points` shape: {fpcl.points.shape} and dtype: {fpcl.points.dtype}".format(fpcl=fpcl))
print("`.mask` shape: {fpcl.mask.shape} and dtype: {fpcl.mask.dtype}".format(fpcl=fpcl))
print("`.colors` shape: {fpcl.colors.shape} and dtype: {fpcl.colors.dtype}".format(fpcl=fpcl))
print("`.normals` shape: {fpcl.normals.shape} and dtype: {fpcl.normals.dtype}".format(fpcl=fpcl))
print("")

##
# Import `geoshow` for displaying interactive 3D of the geometries.

from fiontb.viz import geoshow
geoshow([fpcl], title="FramePointCloud")

##
# A loose point cloud with no specific ordering of its points

from fiontb.pointcloud import PointCloud
# PointCloud also have a `from_frame` method
pcl = fpcl.unordered_point_cloud(world_space=True)
print("PointCloud:")
print("`.points` shape: {pcl.points.shape} and dtype: {pcl.points.dtype}".format(pcl=pcl))
print("`.colors` shape: {pcl.colors.shape} and dtype: {pcl.colors.dtype}".format(pcl=pcl))
print("`.normals` shape: {pcl.normals.shape} and dtype: {pcl.normals.dtype}".format(pcl=pcl))
print("")
geoshow([fpcl], title="PointCloud")

##
# "Surfels are glorified point clouds". Besides the common point cloud
# attributes, surfels also have radius and a confidence scale number.
from fiontb.surfel import SurfelCloud
scl = SurfelCloud.from_frame(frame)
print("`.points` shape: {scl.points.shape} and dtype: {scl.points.dtype}".format(scl=scl))
print("`.colors` shape: {scl.colors.shape} and dtype: {scl.colors.dtype}".format(scl=scl))
print("`.normals` shape: {scl.normals.shape} and dtype: {scl.normals.dtype}".format(scl=scl))
print("`.radii` shape: {scl.radii.shape} and dtype: {scl.radii.dtype}".format(scl=scl))
print("`.confidences` shape: {scl.confidences.shape} and dtype: {scl.confidences.dtype}".format(scl=scl))

geoshow([scl], title="SurfelCloud")
```

## Registration

ICP Odometry

```python
##
# Lets use the sample1 as our frame source.
from fiontb.testing import load_sample1_dataset
dataset = load_sample1_dataset()

# Grab some frames
frame0 = dataset[0]
frame1 = dataset[14]

##
# Pass bilateral filter to reduce depth noise
from fiontb.processing import bilateral_depth_filter
frame0.depth_image = bilateral_depth_filter(frame0.depth_image)
frame1.depth_image = bilateral_depth_filter(frame1.depth_image)

##
# Create an instance of ICP algorithm. Using multiscaling
from fiontb.registration.icp import MultiscaleICPOdometry, ICPOption, ICPVerifier
from fiontb.processing import to_color_feature
icp = MultiscaleICPOdometry([ICPOption(1.0, iters=10), # <- Scales are listed in the inverse order that they're applied
                             ICPOption(0.5, iters=15),
                             ICPOption(0.5, iters=20)])

# Run the algorithm. The overload accepts `fiontb.frame.Frame` or `fiontb.frame.FramePointCloud`.
result = icp.estimate_frame(frame1, frame0,
                            source_feats=to_color_feature(frame1.rgb_image),
                            target_feats=to_color_feature(frame0.rgb_image))

# Simple verifier routine, uses the matching ratio of points and max convariance
verifier = ICPVerifier(match_ratio_threshold=6e-1, covariance_max_threshold=1e-04)
if not verifier(result):
    print("Bad result detected")

##
# Show the result
from fiontb.viz import geoshow
from fiontb.pointcloud import PointCloud

geoshow([PointCloud.from_frame(frame0, world_space=False)[0],
         PointCloud.from_frame(frame1, world_space=False)[0].transform(result.transform.float())],
        title="Aligned frames", invert_y=True)

##
# Evaluate the result
from fiontb.metrics.trajectory import relative_translational_error, relative_rotational_error
from fiontb.camera import RTCamera

gt_trajectory = {frame0.info.timestamp: frame0.info.rt_cam,
                 frame1.info.timestamp: frame1.info.rt_cam}
pred_trajectory = {frame0.info.timestamp: RTCamera(),
                   frame1.info.timestamp: RTCamera(result.transform)}
print("Translational error: ",
      relative_translational_error(gt_trajectory, pred_trajectory).item())
print("Rotational error: ",
      relative_rotational_error(gt_trajectory, pred_trajectory).item())
```

ICP point cloud alignment:

```python
pcl0 = PointCloud.from_frame(dataset[0])
pcl1 = PointCloud.from_frame(dataset[19])

icp.estimate_pcl(pcl0)
icp.estimate_pcl(pcl1)
geoshow(result)
```

Evaluation of trajectory:

```python

```

Surfel fusion:

```python

SurfelModel
```

