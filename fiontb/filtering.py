"""Common filtering of 3d reconstruction for frames.
"""

import numpy as np
import torch

from fiontb._cfiontb import (bilateral_filter_depth_image
                             as _bilateral_filter_depth_image)
from fiontb._utils import ensure_torch


def bilateral_filter_depth_image(depth, mask, filter_width=6,
                                 sigma_d=4.50000000225,
                                 sigma_r=29.9999880000072,
                                 depth_scale=1.0):
    return _bilateral_filter_depth_image(ensure_torch(depth),
                                         ensure_torch(mask, dtype=torch.uint8),
                                         filter_width, sigma_d, sigma_r, depth_scale)


def _valid_projs(uv, width, height):
    return ((uv[:, 0] >= 0) & (uv[:, 0] < width)
            & (uv[:, 1] >= 0) & (uv[:, 1] < height))


class ImageGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, uv):
        valid_uvs = _valid_projs(uv, image.size(1), image.size(0))
        uv = uv[valid_uvs, :]
        uv = uv.long()

        ctx.save_for_backward(image, uv, valid_uvs)
        return image[uv[:, 1], uv[:, 0]].squeeze(), valid_uvs

    @staticmethod
    def backward(ctx, dy_image, dy_uv):

        image, uv, valid_uvs = ctx.saved_tensors
        dtype = dy_image.dtype
        device = dy_image.device
        convx = torch.nn.Conv2d(1, 1, 3, stride=1, padding=1, bias=False)
        convx.weight = torch.nn.Parameter(torch.tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=dtype, device=device).view(1, 1, 3, 3))

        convy = torch.nn.Conv2d(1, 1, 3, stride=1, padding=1, bias=False)
        convy.weight = torch.nn.Parameter(torch.tensor(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=dtype, device=device).view(1, 1, 3, 3))

        xgrad = convx(image.unsqueeze(0).unsqueeze(0)).squeeze().squeeze()
        ygrad = convy(image.unsqueeze(0).unsqueeze(0)).squeeze().squeeze()

        uv2 = torch.zeros(valid_uvs.size(0), 2, device=device)
        # import ipdb; ipdb.set_trace()

        xgrad = xgrad[uv[:, 1], uv[:, 0]]
        ygrad = ygrad[uv[:, 1], uv[:, 0]]
        valid_uvs = valid_uvs.detach()
        uv2[valid_uvs, 0] = xgrad
        uv2[valid_uvs, 1] = ygrad

        return None, uv2


class _Tests:
    def bilateral(self):
        from pathlib import Path
        import cv2
        import matplotlib.pyplot as plt

        depth = cv2.imread(str(Path(__file__).parent / "_tests/assets" / "frame_depth.png"),
                           cv2.IMREAD_ANYDEPTH)
        mask = cv2.imread(str(Path(__file__).parent / "_tests/assets" / "frame_mask.png"),
                          cv2.IMREAD_ANYDEPTH)

        plt.figure()
        plt.title("input")
        plt.imshow(depth)
        filter_depth = bilateral_filter_depth_image(
            depth.astype(np.int32), depth > 0,
            13, 4.50000000225,
            29.9999880000072)

        filtered_depth_image = cv2.bilateralFilter(
            depth.astype(np.float32),
            13, 4.50000000225,
            29.9999880000072)
        plt.figure()
        plt.title("cv2")
        plt.imshow(filtered_depth_image)

        plt.figure()
        plt.title("bilateral")
        plt.imshow(filter_depth)

        plt.show()

    def image_gradient(self):
        from pathlib import Path

        import cv2

        from fiontb.data.ftb import load_ftb

        frame = load_ftb(Path(__file__).parent / "./pose/_test/sample2")[0]
        
        image = torch.from_numpy(cv2.cvtColor(
            frame.rgb_image, cv2.COLOR_RGB2GRAY)).float() / 255.0

        image_op = ImageGradient.apply
        uv = torch.tensor(
            [[45.5, 23.2],
             [52.3, 21.2]], requires_grad=True)
        values, valid = image_op(image, uv)

        import ipdb; ipdb.set_trace()

        values.backward((torch.Tensor([1, 1]), None))
        print(uv.grad)
        
if __name__ == '__main__':
    import fire

    fire.Fire(_Tests)
