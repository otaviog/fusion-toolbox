import rflow

from fusionkit.data.iclnuim import load_icl_nuim
from fusionkit.viz.datasetviewer import DatasetViewer

import numpy as np
import ipyvolume as ipv
from mayavi import mlab


class View(rflow.Interface):
    def evaluate(self):
        innet_traj = load_icl_nuim('living_room')

        viewer = DatasetViewer(innet_traj)
        viewer.run()


class TSDF(rflow.Interface):
    def evaluate(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from fusionkit.tsdf import TSDFFusion
        from fusionkit.camera import RTCamera

        traj = load_icl_nuim('living_room')
        tsdf = TSDFFusion((256, 256, 256), 0.05)

        snap = traj[0]
        mtx = np.eye(4)
        tsdf.update(snap.depth_image, snap.kcam, RTCamera(mtx))

        mlab.pipeline.volume(mlab.pipeline.scalar_field(tsdf.volume))
        mlab.show()

        plt.show()


@rflow.graph()
def view(g):
    g.run = View()


@rflow.graph()
def tsdf(g):
    g.run = TSDF()
