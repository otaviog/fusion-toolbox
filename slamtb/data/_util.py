class TransformCameraDataset:
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

        info0 = dataset.get_info(0)
        if info0.rt_cam is None:
            raise RuntimeError("Dataset does not have camera information")

        self.base = info0.rt_cam.matrix.inverse()

    def __getitem__(self, idx):
        frame = self.dataset[idx]
        frame.info = frame.info.clone()
        frame.info.rt_cam.matrix = self.transform @ frame.info.rt_cam.matrix
        return frame

    def __len__(self):
        return len(self.dataset)

    def get_info(self, idx):
        info = self.dataset.get_info(idx).clone()
        info.rt_cam.matrix = self.transform @ info.rt_cam.matrix
        return info


def set_start_at_eye(dataset):
    info0 = dataset.get_info(0)
    if info0.rt_cam is None:
        raise RuntimeError("Dataset does not have camera information")

    base = info0.rt_cam.matrix.inverse()

    return TransformCameraDataset(dataset, base)
