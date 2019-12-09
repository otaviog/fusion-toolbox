"""Rflow nodes for processing point clouds in workflows.
"""

from pathlib import Path
import json

import torch
import rflow


def _write_transform_to_json(filepath, transform):
    with open(filepath, 'w') as file:
        json.dump(
            {'transformation': transform.tolist()}, file)


def _read_transfrom_from_json(filepath):
    with open(filepath, 'r') as file:
        transform = torch.tensor(
            json.load(file)['transformation']).reshape(4, 4)
    return transform


class InitialAlign(rflow.Interface):
    def evaluate(self, fixed_pcl, mov_pcl,
                 initial_matrix):
        if False:
            mov_center = mov_pcl.verts.mean(0)
            fix_center = fixed_pcl.points.mean(0)

            center_trans = torch.eye(4)
            center_trans[:3, 3] = fix_center - mov_center

        transform = initial_matrix

        mov_pcl = mov_pcl.transform(transform.float())

        return mov_pcl


class ManualAlign(rflow.Interface):
    def evaluate(self, resource, fixed_geo, mov_pcl):
        from .viz import AlignTool

        align_tool = AlignTool(fixed_geo, mov_pcl)
        align_tool.run()

        _write_transform_to_json(resource.filepath, align_tool.transformation)

        mov_pcl.itransform(align_tool.transformation)
        return mov_pcl

    def load(self, resource, mov_pcl):
        transformation = _read_transfrom_from_json(resource.filepath)
        mov_pcl.itransform(transformation)
        return mov_pcl


class LocalRegistration(rflow.Interface):
    """Use the Open3D ICP algorithm to align two point clouds. """

    def evaluate(self, resource, fixed_pcl, mov_pcl, threshold=0.02):
        import open3d
        import numpy as np

        fixed_pcl = fixed_pcl.to_open3d()
        mov_pcl_o3d = mov_pcl.to_open3d()

        trans_init = np.eye(4)

        if False:  # TODO: test if both inputs have colors
            result = open3d.registration.registration_colored_icp(
                mov_pcl_o3d, fixed_pcl, 0.5, np.eye(4),
                open3d.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                           relative_rmse=1e-6, max_iteration=1000))
        else:
            result = open3d.registration.registration_icp(
                mov_pcl_o3d, fixed_pcl, threshold, trans_init,
                open3d.registration.TransformationEstimationPointToPlane(),
                open3d.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-8,
                    relative_rmse=1e-8, max_iteration=1000))

        transform = torch.from_numpy(result.transformation).float()
        _write_transform_to_json(resource.filepath, transform)

        mov_pcl.itransform(transform)

        return mov_pcl

    def load(self, resource, mov_pcl):
        transformation = _read_transfrom_from_json(resource.filepath)
        mov_pcl.itransform(transformation)
        return mov_pcl


class ViewAlignment(rflow.Interface):
    """View two point clouds.
    """

    def evaluate(self, fixed_geo, mov_geo):
        """
        Args:

            fixed_model_path: fixed model path.

            mov_model_path: moveable model path.
        """
        from .viz import ReconstructionViewer

        viewer = ReconstructionViewer(fixed_geo, mov_geo)
        viewer.run()


class ChamferMetric(rflow.Interface):
    """Calculates the Chamfer distance between two point clouds
    """

    def evaluate(self, rec_pcl, gt_pcl):
        """Args:

        source_path: reconstruction point cloud file path.

        gt_path: ground truth point cloud file path
        """
        from fiontb.metrics import chamfer_score

        score = chamfer_score(rec_pcl.points, gt_pcl.points)
        result = {"Chamfer": score}
        self.save_measurement(result)
        print(result)


class NearestPoints(rflow.Interface):
    def evaluate(self, resource, rec_pcl, gt_mesh):
        import torch
        import tenviz

        from fiontb.metrics.mesh import mesh_accuracy
        from fiontb.spatial.trigoctree import TrigOctree

        faces = tenviz.geometry.to_triangles(gt_mesh.faces)
        tree = TrigOctree(gt_mesh.verts,
                          faces.long(), 1024)
        gt_points, _ = tree.query_closest_points(rec_pcl.points)

        torch.save(gt_points, resource.filepath)
        return gt_points

    def load(self, resource):
        import torch
        return torch.load(resource.filepath)


class AccuracyMetric(rflow.Interface):
    def evaluate(self, rec_pcl, nearest_gt_points, thresh_dist):
        from fiontb.metrics.mesh import mesh_accuracy

        score = mesh_accuracy(rec_pcl.points, nearest_gt_points, thresh_dist)
        result = {"Mesh Accuracy": score}
        self.save_measurement(result)
        print(result)


class HeatMapMesh(rflow.Interface):
    def evaluate(self, resource, nearest_points, rec_pcl):
        from matplotlib.pyplot import get_cmap
        from matplotlib.colors import Normalize
        import numpy as np

        from tenviz.io import write_3dobject

        distances = (rec_pcl.points - nearest_points).norm(dim=1)
        distances = np.array(Normalize()(distances.numpy()))

        cmap = get_cmap('plasma', distances.size)
        colors = cmap(distances)

        colors = (colors[:, :3]*255).astype(np.uint8)
        rec_pcl.colors = torch.from_numpy(colors)
        write_3dobject(resource.filepath, rec_pcl.points, normals=rec_pcl.normals,
                       colors=colors)

        return rec_pcl

    def load(self, resource):
        from tenviz.io import read_3dobject

        return read_3dobject(resource.filepath).torch()


def create_evaluation_graph(sub, rec_node, gt_mesh_node, gt_pcl_node,
                            output_prefix, init_matrix=None,
                            show_only_end_nodes=True):
    sub.start_view = ViewAlignment()
    with sub.start_view as args:
        args.fixed_geo = gt_pcl_node
        args.mov_geo = rec_node
    sub.start_view.show = not show_only_end_nodes

    manual_align_mov = rec_node
    if init_matrix is not None:
        sub.init_align = InitialAlign()
        sub.init_align.show = False
        with sub.init_align as args:
            args.fixed_pcl = gt_pcl_node
            args.mov_pcl = rec_node
            args.initial_matrix = init_matrix

        manual_align_mov = sub.init_align

        sub.init_align_view = ViewAlignment()
        sub.init_align_view.show = not show_only_end_nodes
        with sub.init_align_view as args:
            args.fixed_geo = gt_pcl_node
            args.mov_geo = sub.init_align

    sub.manual_align = ManualAlign(rflow.FSResource(
        "{}manual-align.json".format(output_prefix)))
    sub.manual_align.show = not show_only_end_nodes
    with sub.manual_align as args:
        args.fixed_geo = gt_mesh_node
        args.mov_pcl = manual_align_mov

    sub.manual_align_view = ViewAlignment()
    sub.manual_align_view.show = not show_only_end_nodes
    with sub.manual_align_view as args:
        args.fixed_geo = gt_mesh_node
        args.mov_geo = sub.manual_align

    sub.local_reg = LocalRegistration(rflow.FSResource(
        "{}local-reg.json".format(output_prefix)))
    sub.local_reg.show = not show_only_end_nodes
    with sub.local_reg as args:
        args.fixed_pcl = gt_pcl_node
        args.mov_pcl = sub.manual_align

    sub.local_reg_view = ViewAlignment()
    sub.local_reg_view.show = not show_only_end_nodes
    with sub.local_reg_view as args:
        args.fixed_geo = gt_pcl_node
        args.mov_geo = sub.local_reg[0]

    sub.final_align_view = ViewAlignment()
    sub.final_align_view.show = not show_only_end_nodes
    with sub.final_align_view as args:
        args.fixed_geo = gt_mesh_node
        args.mov_geo = sub.local_reg[0]

    sub.chamfer = ChamferMetric()
    with sub.chamfer as args:
        args.gt_pcl = gt_pcl_node
        args.rec_pcl = sub.local_reg[0]

    sub.nearest_points = NearestPoints(rflow.FSResource(
        "{}nearest.ply".format(output_prefix)))
    sub.nearest_points.show = not show_only_end_nodes
    with sub.nearest_points as args:
        args.rec_pcl = sub.local_reg[0]
        args.gt_mesh = gt_mesh_node

    sub.accuracy = AccuracyMetric()
    with sub.accuracy as args:
        args.nearest_gt_points = sub.nearest_points
        args.rec_pcl = sub.local_reg[0]
        args.thresh_dist = 0.05

    sub.heat_map = HeatMapMesh(rflow.FSResource(
        "{}-hmap.ply".format(output_prefix)))
    sub.heat_map.show = not show_only_end_nodes
    with sub.heat_map as args:
        args.nearest_points = sub.nearest_points
        args.rec_pcl = sub.local_reg[0]

    sub.heat_map_view = ViewAlignment()
    with sub.heat_map_view as args:
        args.fixed_geo = gt_pcl_node
        args.mov_geo = sub.heat_map
