from fusionkit.data.sunrgbd import load_sunrgbd_scene
from fusionkit.viz.datasetviewer import DatasetViewer


def main():
    dataset = load_sunrgbd_scene('SUNRGBD/kv1/NYUdata')

    viewer = DatasetViewer(dataset)
    viewer.run()


if __name__ == '__main__':
    main()
