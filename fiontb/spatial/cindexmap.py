import fiontb.fiontblib


class cIndexMap(fiontb.fiontblib.IndexMap):
    def __init__(self, model_points, kcam, img_width, img_height, map_scale):
        proj_points = kcam.project_and_cull(model_points, img_width, img_height)
        super(cIndexMap, self).__init__(proj_points,
                                        int(img_width*map_scale), int(img_height*map_scale))
