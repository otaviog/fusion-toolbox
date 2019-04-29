"""Workflow for generate a FTB version of a SceneNN scene.
"""

import rflow

# pylint: disable=missing-docstring,no-self-use


class ToFTB(rflow.Interface):
    def evaluate(self, resource, oni_fsres, cam_traj, max_frames):
        from fiontb.data.scenenn import load_scenenn
        from fiontb.data.ftb import write_ftb, load_ftb

        scenenn = load_scenenn(oni_fsres.filepath, cam_traj.filepath)
        write_ftb(resource.filepath, scenenn, max_frames=max_frames)

        return load_ftb(resource.filepath)

    def load(self, resource):
        from fiontb.data.ftb import load_ftb
        return load_ftb(resource.filepath)


class View(rflow.Interface):
    def evaluate(self, dataset):
        from fiontb.viz.datasetviewer import DatasetViewer

        viewer = DatasetViewer(dataset)
        viewer.run()


@rflow.graph()
def sample(g):
    g.oni = rflow.shell.ShowMessageIfNotExists(
        rflow.FSResource("045.oni"))
    g.oni.args.message = "Please, download the 045.oni scene from SceneNN"
    
    g.traj = rflow.shell.ShowMessageIfNotExists(
        rflow.FSResource("trajectory.log"))
    g.traj.args.message = "Please, also find the 045.oni trajectory from SceneNN"
    
    g.to_ftb = ToFTB(rflow.FSResource("ftb"))
    with g.to_ftb as args:
        args.oni_fsres = g.oni.resource
        args.cam_traj = g.traj.resource
        args.max_frames = 100

    g.view_ftb = View()
    g.view_ftb.args.dataset = g.to_ftb


if __name__ == '__main__':
    rflow.command.main()
