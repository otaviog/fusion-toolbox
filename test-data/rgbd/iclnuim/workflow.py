from pathlib import Path

import rflow


class LoadICLNUIM(rflow.Interface):
    def evaluate(self, resource, obj_gt_filepath):
        from fiontb.data.iclnuim import load_icl_nuim

        dataset = load_icl_nuim(
            resource.filepath,
            ground_truth_model_path=obj_gt_filepath)

        return dataset


class ToFTB(rflow.Interface):
    def evaluate(self, resource, dataset, max_frames):
        from fiontb.data.ftb import write_ftb

        write_ftb(resource.filepath, dataset, max_frames=max_frames)

        return self.load(resource)

    def load(self, resource):
        from fiontb.data.ftb import load_ftb
        return load_ftb(resource.filepath)


@rflow.graph()
def iclnuim(g):
    """ICL-NUIM dataset loading
    """

    in_base_path = Path('ICL-NUIM')

    scenes = {
        'lr0': ('living_room_traj0_loop', 'living-room.obj'),
        'lr1': ('living_room_traj1_loop', 'living-room.obj')
    }

    for scene_id, (scene_path, obj_file) in scenes.items():
        with g.prefix(scene_id + '_') as sub:
            gt_mesh_filepath = in_base_path / obj_file

            sub.load = LoadICLNUIM(rflow.FSResource(
                in_base_path / scene_path))
            sub.load.args.obj_gt_filepath = gt_mesh_filepath.absolute()

            sub.to_ftb = ToFTB(rflow.FSResource(scene_id))
            with sub.to_ftb as args:
                args.dataset = sub.load
                args.max_frames = 100
