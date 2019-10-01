"""Surfel space carving for removing outliers. More info on: Keller,
Maik, Damien Lefloch, Martin Lambers, Shahram Izadi, Tim Weyrich, and
Andreas Kolb. "Real-time 3d reconstruction in dynamic scenes using
point-based fusion." In 2013 International Conference on 3D Vision-3DV
2013, pp. 1-8. IEEE, 2013.
"""

from pathlib import Path

import torch

from .indexmap import ModelIndexMap
from ._ckernels import surfel_cave_free_space


class SpaceCarving:
    """Remove unstable surfels in front of recently updated stable
    surfels.

    """

    def __init__(self, surfel_model, stable_conf_thresh,
                 search_size=2, min_z_difference=0.1, use_cpu=False):
        """
        Args:
            surfel_model
             (:obj:`fiontb.fusion.surfel.model.SurfelModel`): Surfel model.
        """

        self.surfel_model = surfel_model
        self.stable_conf_thresh = stable_conf_thresh
        self.search_size = search_size
        self.min_z_difference = min_z_difference

        self.use_cpu = use_cpu

    def carve(self, model_indexmap, proj_matrix, rt_cam, width, height,
              curr_time, debug=False):
        """
        Args:

            proj_matrix (:obj:`torch.Tensor`):

        """

        model_indexmap.show_debug("Model", debug)

        context = self.surfel_model.context
        with context.current():
            model_positions = model_indexmap.position_confidence_tex.to_tensor()
            model_idxs = model_indexmap.index_tex.to_tensor()

        if self.use_cpu:
            model_positions = model_positions.cpu()
            model_idxs = model_idxs.cpu()

        surfel_cave_free_space(model_positions, model_idxs,
                               self.surfel_model.active_mask,
                               curr_time, self.stable_conf_thresh,
                               self.search_size, self.min_z_difference)
