from pathlib import Path

import fire

from slamtb.data.ftb import load_ftb
from slamtb.ui import FrameUI, SurfelReconstructionUI, RunMode
from slamtb.sensor import DatasetSensor
from slamtb._utils import profile
from slamtb.testing import get_color_feature, ColorMode

from ..surfel_slam import SurfelSLAM


_TEST_DATA = Path(__file__).parent / "../../../test-data/rgbd"


def _run_test(dataset):
    slam = SurfelSLAM(tracking_mode='frame-to-model', feature_size=1)

    sensor_ui = FrameUI("Frame Control")
    rec_ui = SurfelReconstructionUI(slam.model, RunMode.STEP,
                                    stable_conf_thresh=slam.fusion.stable_conf_thresh)

    sensor = DatasetSensor(dataset)

    with profile(Path(__file__).parent / "surfel_slam.prof"):
        for _ in rec_ui:
            frame = sensor.next_frame()
            if frame is None:
                break

            sensor_ui.update(frame)
            stats = slam.step(frame, features=get_color_feature(
                frame.rgb_image, blur=False, color_mode=ColorMode.GRAY))
            print("Frame {} - {}".format(rec_ui.frame_count, stats))


class Tests:
    def real_scene(self):
        dataset = load_ftb(_TEST_DATA / "sample1")
        _run_test(dataset)

    def synthetic_scene(self):
        dataset = load_ftb(_TEST_DATA / "sample2")
        _run_test(dataset)


if __name__ == '__main__':
    fire.Fire(Tests)
