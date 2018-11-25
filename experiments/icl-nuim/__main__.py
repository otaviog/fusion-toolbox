from fusionkit.data.iclnuim import load_icl_nuim
from fusionkit.viz.datasetviewer import DatasetViewer


def main():
    innet_traj = load_icl_nuim('living_room')

    viewer = DatasetViewer(innet_traj)
    viewer.run()


if __name__ == '__main__':
    main()
