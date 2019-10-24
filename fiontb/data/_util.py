class StartAtEyeDataset:
    def __init__(self, dataset):
        self.dataset = dataset

        info0 = dataset.get_info(0)
        if info0.rt_cam is None:
            raise RuntimeError("Dataset does not have camera information")

        self.base = info0.rt_cam.matrix.inverse()

    def __getitem__(self, idx):
        frame = self.dataset[idx]
        frame.info.rt_cam.matrix = self.base @ frame.info.rt_cam.matrix
        return frame

    def __len__(self):
        return len(self.dataset)

    def get_info(self, idx):
        info = self.dataset.get_info(idx)
        info.rt_cam.matrix = self.base @ info.rt_cam.matrix
        return info


def set_cameras_to_start_at_eye(dataset):
    return
    info0 = dataset.get_info(0)
    if info0.rt_cam is None:
        raise RuntimeError("Dataset does not have camera information")w

    base = info0.rt_cam.matrix.inverse()

    for i in range(len(dataset)):
        infoi = dataset.get_info(i)
        infoi.rt_cam.matrix = base @ infoi.rt_cam.matrix
        dataset.set_info(i, infoi)
