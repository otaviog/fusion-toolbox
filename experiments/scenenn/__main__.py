from fusionkit.data.scenenn import load_scenenn
from fusionkit.viz.datasetviewer import DatasetViewer


def main():
    innet_traj = load_scenenn("030/030.oni", "030/trajectory.log", "asus")

    viewer = DatasetViewer(innet_traj)
    viewer.run()


if __name__ == '__main__':
    main()
