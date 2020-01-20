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
