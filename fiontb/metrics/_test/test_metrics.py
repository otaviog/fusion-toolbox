"""Test metrics
"""

import unittest
from pathlib import Path

import torch
import tenviz

import fiontb.metrics
from fiontb.data.tumrgbd import read_trajectory
from fiontb.spatial.trigoctree import TrigOctree


class TestGeometryMetrics(unittest.TestCase):
    """Test the metrics for comparing geometric reconstructions.
    """

    def test_reconstruction_accuracy(self):
        bunny_root = Path(__file__).parent / "../../../test-data/bunny"

        hq_verts, hq_trigs = tenviz.io.read_3dobject(
            bunny_root / 'bun_zipper.ply')
        lq4_verts, _ = tenviz.io.read_3dobject(
            bunny_root / 'bun_zipper_res4.ply')

        octree = TrigOctree(hq_verts, hq_trigs.long(), 200)

        torch.manual_seed(10)
        for lq_verts, noise, expected in [
                (lq4_verts, 0.01, 0.9249448180198669),
                (lq4_verts, 0.05, 0.18101544678211212),
                (lq4_verts, 0.1, 0.011037527583539486)]:
            lq_verts += torch.rand(*lq_verts.shape, dtype=torch.float32)*noise
            lq_closest, _ = octree.query_closest_points(lq_verts)

            acc = fiontb.metrics.reconstruction_accuracy(lq_verts, lq_closest)
            self.assertAlmostEqual(expected, acc)


class TestTrajectoryMetrics(unittest.TestCase):
    """Test the metrics for comparing trajectories.
    """
    @classmethod
    def setUpClass(cls):
        traj_root = Path(__file__).parent / "../../../test-data/trajectory/"

        cls.gt_traj = read_trajectory(
            traj_root / "freiburd1_desk_gt.traj")
        cls.pred_traj = read_trajectory(
            traj_root / "freiburd1_desk_pred.traj")

    def test_ate(self):
        """Absolute Translation Error.
        """
        ate = fiontb.metrics.absolute_translational_error(
            self.gt_traj, self.pred_traj)
        self.assertAlmostEqual(0.5349, ate.mean().sqrt().item(), places=4)

    def test_rte(self):
        """Relative Translation Error.
        """
        rte = fiontb.metrics.relative_translational_error(
            self.gt_traj, self.pred_traj)
        self.assertAlmostEqual(0.1008, rte.mean().sqrt().item(), places=4)

    def test_are(self):
        """Absolute Rotational Error.
        """
        are = fiontb.metrics.absolute_rotational_error(
            self.gt_traj, self.pred_traj)
        self.assertAlmostEqual(0.1438, are.mean().sqrt().item(), places=4)

    def test_rre(self):
        """Relative Rotational Error.
        """
        rre = fiontb.metrics.relative_rotational_error(
            self.gt_traj, self.pred_traj)
        self.assertAlmostEqual(0.0069, rre.mean().sqrt().item(), places=4)


if __name__ == "__main__":
    unittest.main()
