# Fusion Toolbox

A framework to build RGBD simultaneous localization and mapping (SLAM) algorithms in PyTorch.

This work was sponsered by Eldorado Research Institute
## Features

Dataset parsing:

```python
from fiontb.dataset.tumrgbd import load_tumrgbd

dataset = load_tumrgbd("scene1")

frame0 = dataset[0]
print(frame)

print(frame.depth_image.shape)
print(frame.rgb_image.shape)
```

Visualization:

```python
from fiontb.viz import geoshow

geoshow([frame0])


from fiontb.viz.datasetviewer import DatasetViewer

DatasetViewer(dataset).run()
```

Data structures:

```python
from fiontb.pointcloud import PointCloud


from fiontb.surfel import SurfelCloud
```


Processing:

```python
from fiontb.processing
```

ICP-based odometry:

```python

from fiontb.registration import MultiScaleICP

```

ICP-based point cloud alignment:

```python
```

Surfel fusion:

```python
```

Evaluation:

```python

```
