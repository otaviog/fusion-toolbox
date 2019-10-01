
def set_cameras_to_start_at_eye(dataset):
    info0 = dataset.get_info(0)
    if info0.rt_cam is None:
        raise RuntimeError("Dataset does not have camera information")

    base = info0.rt_cam.matrix.inverse()

    for i in range(len(dataset)):
        infoi = dataset.get_info(i)
        infoi.rt_cam.matrix = base @ infoi.rt_cam.matrix
        dataset.set_info(i, infoi)
