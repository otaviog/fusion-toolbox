"""Unit testing utilities.
"""

import json
from pathlib import Path

import fiontb.data.klg as klg
from fiontb.data.ftb import kcam_from_json
from fiontb.data.tumrgbd import read_trajectory


def load_scene2():
    """Load the scene test-data/rgbd/scene2.klg"""

    base_path = Path(__file__).parent.parent.parent.parent / "test-data/rgbd"
    dataset = klg.KLG(str(base_path / "scene2.klg"))

    with open(str(base_path / "scene2-kcam.json"), 'r') as file:
        dataset.kcam = kcam_from_json(json.load(file))

    dataset.trajectory = list(read_trajectory(
        base_path / "scene2-rtcam.freiburg").values())

    return dataset
