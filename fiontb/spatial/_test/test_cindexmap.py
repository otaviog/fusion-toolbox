import unittest

import torch
import numpy as np

from fiontb.spatial.cindexmap import cIndexMap
from fiontb.camera import KCamera, RTCamera, Homogeneous

class TestIndexMap(unittest.TestCase):
    def test_query(self):
        torch.manual_seed(10)
        model_points = torch.rand(1000, 3, dtype=torch.float32)

        cam_px_width = 640
        cam_px_height = 480
        
        kcam = KCamera.create_from_params(211, 211, (cam_px_width/2.0, cam_px_height/2.0))
        rt_cam = RTCamera(np.eye(4))
        #rt_cam = rt_cam.translate(0, 0, -1)
        
        model_points = Homogeneous(torch.from_numpy(rt_cam.matrix).float()) @ model_points
        index_map = cIndexMap(model_points, kcam, cam_px_width, cam_px_height, 1.0)

        query_points = model_points + 0.01
        query_points = kcam.project_and_cull(query_points, cam_px_width, cam_px_height)
        result = index_map.query(query_points, 8)
        import ipdb; ipdb.set_trace()
        pass
        
if __name__ == '__main__':
    unittest.main()
