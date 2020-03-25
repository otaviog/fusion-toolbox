#include "processing.hpp"

#include "accessor.hpp"
#include "error.hpp"
#include "kernel.hpp"

namespace slamtb {

template <Device dev>
struct ErodeMaskKernel {
  const typename Accessor<dev, bool, 2>::T in_mask;
  typename Accessor<dev, bool, 2>::T out_mask;

  ErodeMaskKernel(const torch::Tensor &in_mask, torch::Tensor out_mask)
      : in_mask(Accessor<dev, bool, 2>::Get(in_mask)),
        out_mask(Accessor<dev, bool, 2>::Get(out_mask)) {}

  FTB_DEVICE_HOST void operator()(int row, int col) {
    if (!in_mask[row][col]) {
      out_mask[row][col] = false;
      return;
    }

    const int neighborhood[4][2] = {{-1, 0}, {0, 1}, {0, -1}, {0, 1}};
    for (int k = 0; k < 4; ++k) {
      const int krow = row + neighborhood[k][0];
      const int kcol = col + neighborhood[k][1];

      if (krow >= 0 && krow < in_mask.size(0) && kcol >= 0 &&
          kcol < in_mask.size(1)) {
        if (!in_mask[krow][kcol]) {
          out_mask[row][col] = false;
          return;
        }
      }
    }

    out_mask[row][col] = true;
  }
};

void Processing::ErodeMask(const torch::Tensor &in_mask,
                           torch::Tensor out_mask) {
  const auto ref_device = in_mask.device();
  FTB_CHECK_DEVICE(ref_device, out_mask);

  if (ref_device.is_cuda()) {
    ErodeMaskKernel<kCUDA> kernel(in_mask, out_mask);
    Launch2DKernelCUDA(kernel, in_mask.size(1), in_mask.size(0));
  } else {
    ErodeMaskKernel<kCPU> kernel(in_mask, out_mask);
    Launch2DKernelCPU(kernel, in_mask.size(1), in_mask.size(0));
  }
}
}  // namespace slamtb
