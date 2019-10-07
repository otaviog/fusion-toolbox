
from fiontb.surfel import compute_confidences


class ConfidenceCache:
    def __init__(self):
        self.width = -1
        self.height = -1
        self.confidences = None

    def get_confidences(self, frame_pcl):
        fheight, fwidth = frame_pcl.image_points.shape[:2]
        # It doesn't check kcam
        if fheight != self.height or fwidth != self.width:
            self.width = fwidth
            self.height = fheight
            self.confidences = compute_confidences(frame_pcl, no_mask=True)

        return self.confidences[frame_pcl.mask.flatten()]
