from fiontb.data.ftb import load_ftb
dataset = load_ftb("test-data/rgbd/sample2/")
frame = dataset[0]

from fiontb.processing import bilateral_depth_filter
frame.depth_image = bilateral_depth_filter(
    frame.depth_image, frame.depth_image > 0,
    filter_width=6, sigma_color=30, sigma_space=5)
