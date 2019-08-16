"""Rflow nodes for processing point clouds in workflows.
"""
from pathlib import Path

import rflow


class InitialAlign(rflow.Interface):
    def evaluate(self, resource, fixed_pcl, mov_geo,
                 initial_matrix):
        import torch

        from tenviz.io import write_3dobject

        from fiontb.camera import Homogeneous, normal_transform_matrix

        initial_matrix = torch.tensor(initial_matrix)

        mov_center = mov_geo.verts.mean(0)
        fix_center = fixed_pcl.points.mean(0)

        center_trans = torch.eye(4)
        center_trans[:3, 3] = fix_center - mov_center

        transform = center_trans @ initial_matrix
        mov_geo.verts = Homogeneous(transform) @ mov_geo.verts
        mov_geo.normals = (normal_transform_matrix(transform)
                           @ mov_geo.normals.reshape(-1, 3, 1)).squeeze()

        write_3dobject(resource.filepath,
                       mov_geo.verts, colors=mov_geo.colors, normals=mov_geo.normals)

        return mov_geo

    def load(self, resource):
        from tenviz.io import read_3dobject

        return read_3dobject(resource.filepath).torch()


class ManualAlign(rflow.Interface):
    def evaluate(self, resource, fixed_geo, mov_geo):
        from tenviz.io import write_3dobject
        from fiontb.camera import Homogeneous, normal_transform_matrix
        from .viz import AlignTool

        align_tool = AlignTool(fixed_geo, mov_geo)
        align_tool.run()

        mov_geo.verts = Homogeneous(align_tool.transformation) @ mov_geo.verts
        mov_geo.normals = normal_transform_matrix(
            align_tool.transformation) @ mov_geo.normals.reshape(-1, 3, 1)
        mov_geo.normals = mov_geo.normals.squeeze()

        write_3dobject(resource.filepath,
                       mov_geo.verts,
                       colors=mov_geo.colors,
                       normals=mov_geo.normals)
        return mov_geo

    def load(self, resource):
        from tenviz.io import read_3dobject

        return read_3dobject(resource.filepath).torch()


class LocalRegistration(rflow.Interface):
    """Use the Open3D ICP algorithm to align two point clouds. """

    def evaluate(self, resource, fixed_pcl, mov_geo, threshold=0.02):
        import torch
        import numpy as np
        import json

        from open3d import (registration_icp, TransformationEstimationPointToPlane,
                            registration_colored_icp, ICPConvergenceCriteria,
                            KDTreeSearchParamHybrid)

        from tenviz.io import write_3dobject
        from tenviz.geometry import Geometry

        from fiontb.pointcloud import PointCloud

        fixed_pcl = fixed_pcl.to_open3d()
        mov_pcl = PointCloud(mov_geo.verts, colors=mov_geo.colors,
                             normals=mov_geo.normals)
        mov_pcl_o3d = mov_pcl.to_open3d()

        trans_init = np.eye(4)

        if False:  # TODO: test if both inputs have colors
            reg = registration_colored_icp(
                fixed_pcl, mov_pcl_o3d, 0.5, np.eye(4),
                ICPConvergenceCriteria(relative_fitness=1e-6,
                                       relative_rmse=1e-6, max_iteration=100))
        else:
            reg = registration_icp(fixed_pcl, mov_pcl_o3d, threshold, trans_init,
                                   TransformationEstimationPointToPlane())

        transform = np.linalg.inv(reg.transformation)
        transform = torch.from_numpy(transform).float()
        mov_pcl.transform(transform)

        write_3dobject(
            resource[0].filepath, mov_pcl.points, colors=mov_pcl.colors, normals=mov_pcl.normals)

        with open(resource[1].filepath, 'w') as file:
            json.dump(
                {'transformation': transform.tolist()}, file)

        return Geometry(mov_pcl.points, colors=mov_pcl.colors, normals=mov_pcl.normals), transform

    def load(self, resource):
        import json
        import numpy as np

        from tenviz.io import read_3dobject

        with open(resource[1].filepath, 'r') as file:
            transform = np.array(
                json.load(file)['transformation']).reshape(4, 4)

        return read_3dobject(resource[0].filepath).torch(), transform


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

    def evaluate(self, rec_geo, gt_pcl):
        """Args:

        source_path: reconstruction point cloud file path.

        gt_path: ground truth point cloud file path
        """
        from fiontb.metrics import chamfer_score

        score = chamfer_score(rec_geo.verts, gt_pcl.points)
        result = {"Chamfer": score}
        self.save_measurement(result)
        print(result)


class NearestPoints(rflow.Interface):
    def evaluate(self, resource, rec_geo, gt_mesh):
        import torch

        from fiontb.metrics.mesh import mesh_accuracy
        from fiontb.spatial.trigoctree import TrigOctree

        tree = TrigOctree(gt_mesh.verts,
                          gt_mesh.faces.long(), 1024)
        gt_points, _ = tree.query_closest_points(rec_geo.verts)

        torch.save(gt_points, resource.filepath)
        return gt_points

    def load(self, resource):
        import torch
        return torch.load(resource.filepath)


class AccuracyMetric(rflow.Interface):
    def evaluate(self, rec_geo, gt_mesh, thresh_dist):
        from fiontb.metrics.mesh import mesh_accuracy
        from fiontb.spatial.trigoctree import TrigOctree

        tree = TrigOctree(gt_mesh.verts,
                          gt_mesh.faces.long(), 1024)
        gt_points, _ = tree.query_closest_points(rec_geo.verts)

        score = mesh_accuracy(rec_geo.verts, gt_points, thresh_dist)
        result = {"Mesh Accuracy": score}
        self.save_measurement(result)
        print(result)


class HeatMapMesh(rflow.Interface):
    def evaluate(self, resource, nearest_points, rec_geo):
        from matplotlib.pyplot import get_cmap
        from matplotlib.colors import Normalize
        import numpy as np

        import tenviz.io

        distances = (rec_geo.verts - nearest_points).norm(dim=1)
        distances = np.array(Normalize()(distances.numpy()))

        cmap = get_cmap('plasma', distances.size)
        colors = cmap(distances)

        colors = (colors[:, :3]*255).astype(np.uint8)
        rec_geo.colors = colors
        tenviz.io.write_3dobject(resource.filepath, rec_geo.verts, normals=rec_geo.normals,
                                 colors=colors)

        return rec_geo.torch()

    def load(self, resource):
        import tenviz.io

        return tenviz.io.read_3dobject(resource.filepath).torch()


def evaluation_graph(sub, result_node, gt_cad_mesh, gt_pcl, init_mtx=None, show_only_end_nodes=True):
    import torch
    from .processing import LoadMesh

    result_model_filepath = Path(result_node.get_resource().filepath)

    sub.rec_geo = LoadMesh(result_node.get_resource())
    sub.rec_geo.show = False

    sub.unreg_view = ViewAlignment()
    with sub.unreg_view as args:
        args.fixed_geo = gt_pcl
        args.mov_geo = sub.rec_geo
    sub.unreg_view.show = not show_only_end_nodes

    sub.init_align = InitialAlign(rflow.FSResource(
        result_model_filepath.with_suffix(".ireg.ply")))
    sub.init_align.show = False
    with sub.init_align as args:
        args.fixed_pcl = gt_pcl
        args.mov_geo = sub.rec_geo
        if init_mtx is None:
            init_mtx = torch.eye(4)
            init_mtx[0, 0] = -1
            init_mtx[1, 1] = -1
            init_mtx[2, 2] = 1

        args.initial_matrix = init_mtx.tolist()

    sub.init_align_view = ViewAlignment()
    sub.init_align_view.show = not show_only_end_nodes
    with sub.init_align_view as args:
        args.fixed_geo = gt_pcl
        args.mov_geo = sub.init_align

    sub.manual_align = ManualAlign(rflow.FSResource(
        result_model_filepath.with_suffix(".mreg.ply")))
    sub.manual_align.show = not show_only_end_nodes
    with sub.manual_align as args:
        # args.fixed_model_path = gt_pcl
        args.fixed_geo = gt_cad_mesh
        args.mov_geo = sub.init_align

    sub.manual_align_view = ViewAlignment()
    sub.manual_align_view.show = not show_only_end_nodes
    with sub.manual_align_view as args:
        # args.fixed_model_path = gt_pcl
        args.fixed_geo = gt_cad_mesh
        args.mov_geo = sub.manual_align

    sub.local_reg = LocalRegistration(rflow.MultiResource(
        rflow.FSResource(result_model_filepath.with_suffix(".lreg.ply")),
        rflow.FSResource(result_model_filepath.with_suffix(".lreg.json"))))
    sub.local_reg.show = not show_only_end_nodes
    with sub.local_reg as args:
        # args.fixed_model_path = gt_pcl
        args.fixed_pcl = gt_pcl
        args.mov_geo = sub.manual_align

    sub.local_reg_view = ViewAlignment()
    sub.local_reg_view.show = not show_only_end_nodes
    with sub.local_reg_view as args:
        args.fixed_geo = gt_pcl
        args.mov_geo = sub.local_reg[0]

    sub.final_align_view = ViewAlignment()
    sub.final_align_view.show = not show_only_end_nodes
    with sub.final_align_view as args:
        args.fixed_geo = gt_cad_mesh
        args.mov_geo = sub.local_reg[0]

    sub.chamfer = ChamferMetric()
    with sub.chamfer as args:
        args.gt_pcl = gt_pcl
        args.rec_geo = sub.local_reg[0]

    sub.accuracy = AccuracyMetric()
    with sub.accuracy as args:
        args.gt_mesh = gt_cad_mesh
        args.rec_geo = sub.local_reg[0]
        args.thresh_dist = 0.05

    sub.nearest_points = NearestPoints(rflow.FSResource(
        result_model_filepath.with_suffix('.nearest.ply')))
    sub.nearest_points.show = not show_only_end_nodes
    with sub.nearest_points as args:
        args.rec_geo = sub.local_reg[0]
        args.gt_mesh = gt_cad_mesh

    sub.heat_map = HeatMapMesh(rflow.FSResource(
        result_model_filepath.with_suffix('.hmap.ply')))
    sub.heat_map.show = not show_only_end_nodes
    with sub.heat_map as args:
        args.nearest_points = sub.nearest_points
        args.rec_geo = sub.local_reg[0]

    sub.view_heat_map = ViewAlignment()
    with sub.view_heat_map as args:
        args.fixed_geo = gt_pcl
        args.mov_geo = sub.heat_map
