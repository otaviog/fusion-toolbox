import fiontb.fiontblib


class cIndexMap(fiontb.fiontblib.IndexMap):
    def __init__(self, model_points, kcam, img_width, img_height, window_size=7, depth_slots=9):
        proj_points = kcam.project_and_cull(
            model_points, img_width, img_height)
        super(cIndexMap, self).__init__(proj_points, model_points,
                                        img_width, img_height, window_size, depth_slots)
        self.kcam = kcam

    def query(self, query_points, query_k):
        proj_qp = self.kcam.project(query_points)
        return super(cIndexMap, self).query(proj_qp, query_points, query_k)
