"""Test metrics
"""

import unittest
from pathlib import Path

import torch
import tenviz

import slamtb.metrics
from slamtb.data.tumrgbd import read_trajectory
from slamtb.spatial.trigoctree import TrigOctree


class TestGeometryMetrics(unittest.TestCase):
    """Test the metrics for comparing geometric reconstructions.
    """

    def test_reconstruction_accuracy(self):
        """Test the reconstruction accuracy metrics"""
        bunny_root = Path(__file__).parent / "../../../test-data/bunny"

        hq_verts, hq_trigs = tenviz.io.read_3dobject(
            bunny_root / 'bun_zipper.ply')
        lq4_verts, _ = tenviz.io.read_3dobject(
            bunny_root / 'bun_zipper_res4.ply')

        octree = TrigOctree(hq_verts, hq_trigs.long(), 200)

        torch.manual_seed(10)
        for lq_verts, noise, expected in [
                (lq4_verts, 0.01, 0.9249448180198669),
                (lq4_verts, 0.05, 0.2693156599998474),
                (lq4_verts, 0.1, 0.10154525190591812)]:
            lq_verts = lq_verts + \
                torch.rand(*lq_verts.shape, dtype=torch.float32)*noise
            lq_closest, _ = octree.query_closest_points(lq_verts)

            acc = slamtb.metrics.reconstruction_accuracy(lq_verts, lq_closest)
            self.assertAlmostEqual(expected, acc)

        self.assertAlmostEqual(
            0.556291401386261,
            slamtb.metrics.mesh_reconstruction_accuracy(
                hq_verts, hq_trigs,
                lq4_verts + torch.rand(*lq_verts.shape, dtype=torch.float32)*0.02))


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
        ate = slamtb.metrics.absolute_translational_error(
            self.gt_traj, self.pred_traj)
        self.assertAlmostEqual(0.5349, ate.mean().sqrt().item(), places=4)

    def test_rte(self):
        """Relative Translation Error.
        """
        rte = slamtb.metrics.relative_translational_error(
            self.gt_traj, self.pred_traj)
        self.assertAlmostEqual(0.1008, rte.mean().sqrt().item(), places=4)

    def test_are(self):
        """Absolute Rotational Error.
        """
        are = slamtb.metrics.absolute_rotational_error(
            self.gt_traj, self.pred_traj)
        self.assertAlmostEqual(0.497086, are.mean().sqrt().item(), places=4)

    def test_rre(self):
        """Relative Rotational Error.
        """
        rre = slamtb.metrics.relative_rotational_error(
            self.gt_traj, self.pred_traj)
        self.assertAlmostEqual(0.0992, rre.mean().sqrt().item(), places=4)


if __name__ == "__main__":
    unittest.main()
