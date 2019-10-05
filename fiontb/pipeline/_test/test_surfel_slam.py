from pathlib import Path

import fire

from fiontb.data.ftb import load_ftb
from fiontb.ui import FrameUI, SurfelReconstructionUI, RunMode
from fiontb.sensor import DatasetSensor
from fiontb._utils import profile

from ..surfel_slam import SurfelSLAM

_TEST_DATA = Path(__file__).parent / "../../../test-data/rgbd"


def _run_test(dataset):
    slam = SurfelSLAM()

    sensor_ui = FrameUI("Frame Control")
    rec_ui = SurfelReconstructionUI(slam.model, RunMode.STEP, inverse=False)

    sensor = DatasetSensor(dataset)

    with profile(Path(__file__).parent / "surfel_slam.prof"):
        for _ in rec_ui:
            frame = sensor.next_frame()
            if frame is None:
                break

            sensor_ui.update(frame)
            stats = slam.step(frame)
            print(stats)


class Tests:
    def real_scene(self):
        dataset = load_ftb(_TEST_DATA / "sample1")
        _run_test(dataset)

    def synthetic_scene(self):
        dataset = load_ftb(_TEST_DATA / "sample2")
        _run_test(dataset)


if __name__ == '__main__':
    fire.Fire(Tests)
