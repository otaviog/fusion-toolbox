import rflow


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


@rflow.graph()
def sample(g):
    g.oni = rflow.shell.ShowMessageIfNotExists(
        rflow.FSResource("045.oni"))
    g.traj = rflow.shell.ShowMessageIfNotExists(
        rflow.FSResource("trajectory.log"))

    g.dataset = ToFTB(rflow.FSResource("ftb"))
    with g.dataset as args:
        args.oni_fsres = g.oni.resource
        args.cam_traj = g.traj.resource
        args.max_frames = 100


if __name__ == '__main__':
    rflow.command.main()
