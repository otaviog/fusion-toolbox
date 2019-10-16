from pathlib import Path

import fire
import tenviz

from fiontb.data.ftb import load_ftb
from fiontb.surfel import SurfelModel, SurfelCloud

from ..indexmap import SurfelIndexMapRaster, show_indexmap, ModelIndexMapRaster


class Tests:
    def surfel_raster(self):
        test_data = Path(__file__).parent / "../../../../test-data/rgbd"

        dataset = load_ftb(test_data / "sample2")
        frame = dataset[1]

        gl_context = tenviz.Context()
        surfel_model = SurfelModel(gl_context, 640*480*2)
        surfel_model.add_surfels(
            SurfelCloud.from_frame(frame).transform(frame.info.rt_cam.cam_to_world),
            update_gl=True)

        raster = SurfelIndexMapRaster(surfel_model)
        #raster = ModelIndexMapRaster(surfel_model)

        view = gl_context.viewer([raster.program])
        view.show(1)
        raster.raster(frame.info.kcam.get_opengl_projection_matrix(
            0.01, 500.0), frame.info.rt_cam, 640, 480)

        indexmap = raster.to_indexmap()

        show_indexmap(indexmap)


if __name__ == '__main__':
    fire.Fire(Tests)
