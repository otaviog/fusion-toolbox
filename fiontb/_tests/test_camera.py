import unittest

class TestCamera(unittest.TestCase):
    def test_kcamera(self):
        cam = KCamera.create_from_params()

    def test_rtcamera(self):
        cam = RTcamera.create_from_params()
