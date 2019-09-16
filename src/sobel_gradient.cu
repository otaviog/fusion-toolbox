#include "filtering.hpp"

#include "accessor.hpp"
#include "error.hpp"
#include "kernel.hpp"

namespace fiontb {
namespace {

template <Device dev, typename scalar_t>
struct SobelGradientKernel {
  const typename Accessor<dev, scalar_t, 2>::T image;
  typename Accessor<dev, scalar_t, 3>::T grad;

  SobelGradientKernel(const torch::Tensor &image, torch::Tensor grad)
      : image(Accessor<dev, scalar_t, 2>::Get(image)),
        grad(Accessor<dev, scalar_t, 3>::Get(grad)) {}

  FTB_DEVICE_HOST void operator()(int row, int col) {
    const int width = image.size(1);
    const int height = image.size(0);

    if (row >= height || col >= width) return;

    const float Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    const float Gy[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

    const int row_start = max(row - 1, 0);
    const int row_end = min(row + 1, height - 1);

    const int col_start = max(col - 1, 0);
    const int col_end = min(col + 1, width - 1);

    float gx_sum = 0.0f;
    float gy_sum = 0.0f;

    for (int irow = row_start; irow <= row_end; ++irow) {
      const int krow = irow - row + 1;
      const auto image_row = image[irow];
      for (int icol = col_start; icol <= col_end; ++icol) {
        const int kcol = icol - col + 1;
        const float value = image_row[icol];

        gx_sum += value * Gx[krow][kcol];
        gy_sum += value * Gy[krow][kcol];
      }
    }

    grad[row][col][0] = gx_sum;
    grad[row][col][1] = gy_sum;
  }
};

}  // namespace

void CalcSobelGradient(const torch::Tensor image, torch::Tensor out_grad) {
  if (image.is_cuda()) {
    FTB_CHECK(out_grad.is_cuda(), "Expected a cuda tensor");
    AT_DISPATCH_ALL_TYPES(image.scalar_type(), "SobelGradientKernel", [&] {
      SobelGradientKernel<kCUDA, scalar_t> kernel(image, out_grad);
      Launch2DKernelCUDA(kernel, image.size(1), image.size(0));
    });
  } else {
    FTB_CHECK(!out_grad.is_cuda(), "Expected a cpu tensor");
    AT_DISPATCH_ALL_TYPES(image.scalar_type(), "SobelGradientKernel", [&] {
      SobelGradientKernel<kCPU, scalar_t> kernel(image, out_grad);
      Launch2DKernelCUDA(kernel, image.size(1), image.size(0));
    });
  }
}
}  // namespace fiontb