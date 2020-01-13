from fiotb.testing import load_sample2_dataset

dataset = load_sample2_dataset()

frame0 = dataset[0]
frame1 = dataset[1]

frame0.depth_image = bilateral_filter(frame0.depth_image)
frame1.depth_image = bilateral_filter(frame1.depth_image)

