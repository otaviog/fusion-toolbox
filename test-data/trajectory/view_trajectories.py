#!/usr/bin/env python
"""View the trajectories in this directory.
"""

from pathlib import Path

import fire

from fiontb.data.tumrgbd import read_trajectory
from fiontb.viz.trajectoryviewer import TrajectoryViewer


class Views:
    """View the trajectories in this directory
    """
    @staticmethod
    def freiburd1_desk():
        """Shows the freiburd1_desk pair.
        """
        here = Path(__file__).parent

        gt_traj = read_trajectory(here / "freiburd1_desk_gt.traj")
        pred_traj = read_trajectory(here / "freiburd1_desk_pred.traj")

        TrajectoryViewer([gt_traj, pred_traj], colormaps=["Blues", "Reds"]).run()


if __name__ == "__main__":
    fire.Fire(Views)
