"""Common filtering of 3d reconstruction for frames.
"""

import numpy as np
import torch

from fiontb._cfiontb import (bilateral_depth_filter as _bilateral_depth_filter,
                             featuremap_op_forward as _featuremap_op_forward,
                             featuremap_op_backward as _featuremap_op_backward)
from fiontb._utils import ensure_torch, empty_ensured_size


def bilateral_depth_filter(depth, mask, out_tensor=None, filter_width=6,
                           sigma_d=4.50000000225,
                           sigma_r=29.9999880000072,
                           depth_scale=1.0):
    depth = ensure_torch(depth)
    mask = ensure_torch(mask, dtype=torch.bool)
    out_tensor = empty_ensured_size(out_tensor, depth.size(0), depth.size(1),
                                    dtype=depth.dtype, device=depth.device)

    _bilateral_depth_filter(depth, mask, out_tensor,
                            filter_width, sigma_d, sigma_r, depth_scale)

    return out_tensor


class FeatureMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, uv):

        device = image.device
        dtype = image.dtype
        features = torch.empty(image.size(0), uv.size(
            0), dtype=dtype, device=device)
        bound_mask = torch.empty(uv.size(0), dtype=torch.bool, device=device)

        _featuremap_op_forward(image, uv, features, bound_mask)
        ctx.save_for_backward(image, uv)

        return features, bound_mask

    @staticmethod
    def backward(ctx, dl_value, _):  # dy_bound_mask
        device = dl_value.device
        dtype = dl_value.dtype

        image, uv = ctx.saved_tensors
        dl_uv = torch.empty(uv.size(), dtype=dtype, device=device)
        _featuremap_op_backward(image, uv, dl_value, dl_uv)

        return None, dl_uv


class _Tests:
    # pylint: disable=no-self-use
    def bilateral(self):
        """Uses bilateral_depth_filter on a sample image and compares with OpenCV.
        """
        from pathlib import Path
        import cv2
        import matplotlib.pyplot as plt

        depth = torch.from_numpy(cv2.imread(
            str(Path(__file__).parent / "_tests/assets" / "frame_depth.png"),
            cv2.IMREAD_ANYDEPTH).astype(np.int32))
        mask = depth > 0

        plt.figure()
        plt.title("input")
        plt.imshow(depth)
        filter_depth = bilateral_depth_filter(
            depth, mask,
            None, 13, 4.50000000225,
            29.9999880000072)

        filtered_depth_image = cv2.bilateralFilter(
            depth.float().numpy(), 13, 4.50000000225,
            29.9999880000072)
        plt.figure()
        plt.title("cv2")
        plt.imshow(filtered_depth_image)

        plt.figure()
        plt.title("bilateral")
        plt.imshow(filter_depth)

        plt.show()

    def featuremap(self):
        """Uses the FeatureMap to extract an image gradient on X and Y axis.
        """
        from pathlib import Path

        import cv2
        import matplotlib.pyplot as plt

        from fiontb.data.ftb import load_ftb

        image_op = FeatureMap.apply

        frame = load_ftb(Path(__file__).parent / "./pose/_test/sample2")[0]

        image = torch.from_numpy(cv2.cvtColor(
            frame.rgb_image, cv2.COLOR_RGB2GRAY)).float() / 255.0

        ys, xs = torch.meshgrid(torch.arange(image.size(0), dtype=torch.float),
                                torch.arange(image.size(1), dtype=torch.float))
        uv = torch.stack(
            [xs, ys], 2)
        uv = uv.view(-1, 2)
        uv.requires_grad = True

        values, _ = image_op(image.unsqueeze(0), uv)
        grad = torch.ones(1, uv.size(0))
        values.backward((grad, None))

        xgrad = uv.grad[:, 0].view(image.size(0), image.size(1))
        ygrad = uv.grad[:, 1].view(image.size(0), image.size(1))

        plt.figure()
        plt.title("xgrad")
        plt.imshow(xgrad)

        plt.figure()
        plt.title("ygrad")
        plt.imshow(ygrad)
        plt.show()


if __name__ == '__main__':
    import fire

    fire.Fire(_Tests)
