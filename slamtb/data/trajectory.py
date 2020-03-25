import torch

from slamtb.camera import RTCamera

def read_log_file_trajectory(stream):
    trajectory = []
    while True:
        line = stream.readline()
        if line == "":
            break
        curr_entry = []
        for _ in range(4):
            line = stream.readline()
            curr_entry.append([float(elem) for elem in line.split()])
            # cam space to world space
        rt_mtx = torch.tensor(curr_entry, dtype=torch.float)

        assert rt_mtx.shape == (4, 4)
        trajectory.append(RTCamera(rt_mtx))
    return trajectory
