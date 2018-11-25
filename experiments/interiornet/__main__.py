#!/usr/bin/env python

import argparse

import numpy as np
import cv2

from fusionkit.data.interiornet import load_interiornet
from fusionkit.viz.datasetviewer import DatasetViewer

def main():
    innet_traj = load_interiornet('camera_9_3')

    viewer = DatasetViewer(innet_traj)
    viewer.run()

    if False:
        vis = Visualizer()
        vis.create_window()
        vis.add_geometry(pcl)
        while True:
            vis.update_geometry()
            print(vis.poll_events())
            vis.update_renderer()

if __name__ == '__main__':
    main()
