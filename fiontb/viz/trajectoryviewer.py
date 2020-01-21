"""Trajectory viewer.
"""

import matplotlib.pyplot as plt
import torch

import tenviz


from fiontb.camera import KCamera


class TrajectoryViewer:
    """Viewer for camera trajectories only"""

    def __init__(self, trajectories, kcam=None, cam_far=0.1, colormaps=None,
                 title=None):
        """Initialize the viewer.

        Args:

            trajectories (List[Dict[float:
             :obj:`fiontb.camera.RTCamera`]]): List of trajectory
             dictionary, keys are the timestamps, and values are the
             absolute camera's pose.

            kcam (:obj:`fiontb.camera.KCamera`, optional): Intrinsic camera
             frustum shape.

            cam_far (float): Cameras' far plane.

            colormaps (List[str], optional): Matplotlib colormap names
             for coloring each trajectory.

            title (str, optional): Window title.

        """
        self.gl_context = tenviz.Context()
        self._title = title
        if kcam is None:
            kcam = KCamera.from_params(525, 525, (525/2, 525/2))

        self.proj_cam = kcam.get_projection_params(
            cam_far*.1, cam_far)

        self._scene = []

        self._default_cmaps = ["Blues", "Greens", "Reds", "Purples"]
        self._cmap_count = 0

        if colormaps is None:
            colormaps = [None]*len(trajectories)

        cxs = []
        czs = []
        for traj, color in zip(trajectories, colormaps):
            self.add_trajectory(traj, color)

            cxs.extend([rt_cam.center[0].item() for rt_cam in traj.values()])
            czs.extend([rt_cam.center[2].item() for rt_cam in traj.values()])

        with self.gl_context.current():
            axis = tenviz.nodes.create_axis_grid(
                min(min(cxs), min(czs)),
                max(max(cxs), max(czs)), 10)
            self._scene.append(axis)

    def add_trajectory(self, trajectory, colormap=None):
        """Add a trajectory.

        Args:

            trajectory (Dict[float: :obj:`fiontb.camera.RTCamera`]): A trajectory.

            colormap (str): Matplotlib colormap name for coloring the
             trajectory frustums. Each pose will receive a colormap
             gammut.

        """

        if colormap is None:
            colormap = self._default_cmaps[self._cmap_count % len(
                self._default_cmaps)]
            self._cmap_count += 1
        cmap = plt.get_cmap(colormap, len(trajectory))

        with self.gl_context.current():
            for i, rt_cam in enumerate(trajectory.values()):
                color = cmap(i)[:3]
                node = tenviz.nodes.create_virtual_camera(
                    self.proj_cam,
                    rt_cam.opengl_view_cam.inverse().numpy(),
                    color=color)
                self._scene.append(node)

    def run(self):
        """Show viewer window.
        """
        viewer = self.gl_context.viewer(
            self._scene, cam_manip=tenviz.CameraManipulator.WASD)
        if self._title is not None:
            viewer.title = self._title
        while True:
            key = viewer.wait_key(1)
            if key < 0:
                break

            key = chr(key & 0xff)
            if key == '1':
                viewer.view_matrix = torch.tensor(
                    [[1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 1, 0, -5],
                     [0, 0, 0, 1]], dtype=torch.float)
            elif key == '2':
                viewer.view_matrix = torch.tensor(
                    [[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, -1, -5],
                     [0, 0, 0, 1]], dtype=torch.float)


def _test():
    # pylint: disable=import-outside-toplevel
    from fiontb.testing import load_sample2_dataset

    dataset = load_sample2_dataset()
    trajectory = {i: dataset.get_info(i).rt_cam for i in range(len(dataset))}
    viewer = TrajectoryViewer([trajectory])

    viewer.run()


if __name__ == '__main__':
    _test()
