#include "surfel_fusion_common.hpp"

#include <torch/torch.h>

#include "cuda_utils.hpp"
#include "error.hpp"
#include "kernel.hpp"
#include "math.hpp"

namespace fiontb {

namespace {

const int MAX_VIOLANTIONS = 1;

template <Device dev>
struct CarveSpaceKernel {
  const IndexMapAccessor<dev> model;
  typename Accessor<dev, bool, 1>::T free_mask;

  int32_t curr_time;
  float stable_thresh;
  int search_size;
  float min_z_diff;

  CarveSpaceKernel(const IndexMap &model, torch::Tensor free_mask,
                   int32_t curr_time, float stable_thresh, int search_size,
                   float min_z_diff)
      : model(model),
        free_mask(Accessor<dev, bool, 1>::Get(free_mask)),
        curr_time(curr_time),
        stable_thresh(stable_thresh),
        search_size(search_size),
        min_z_diff(min_z_diff) {}

  FTB_DEVICE_HOST void operator()(int row, int col) {
    if (model.empty(row, col)) return;
    if (model.time(row, col) == curr_time &&
        model.confidence(row, col) >= stable_thresh)
      return;

    const float model_z = model.position_confidence[row][col][2];
    const int model_idx = model.index(row, col);

    int violantion_count = 0;

    const int start_row = max(row - search_size, 0);
    const int end_row = min(row + search_size, model.height() - 1);

    const int start_col = max(col - search_size, 0);
    const int end_col = min(col + search_size, model.width() - 1);

    for (int krow = start_row; krow <= end_row; ++krow) {
      for (int kcol = start_col; kcol <= end_col; ++kcol) {
        if (krow == row && kcol == col) continue;
        if (model.empty(krow, kcol)) continue;
        if (model.time(krow, kcol) != curr_time &&
            model.confidence(krow, kcol) < stable_thresh)
          continue;
        const float stable_z = model.position_confidence[krow][kcol][2];
        if (stable_z - model_z > min_z_diff) {
          ++violantion_count;
        }
      }
    }

    if (violantion_count >= MAX_VIOLANTIONS) {
      free_mask[model_idx] = 1;
    }
  }
};
}  // namespace

void SurfelFusionOp::CarveSpace(const IndexMap &model, torch::Tensor free_mask,
                                int curr_time, float stable_thresh,
                                int search_size, float min_z_diff) {
  const auto ref_device = model.get_device();
  model.CheckDevice(ref_device);

  FTB_CHECK_DEVICE(ref_device, free_mask);

  if (ref_device.is_cuda()) {
    CarveSpaceKernel<kCUDA> kernel(model, free_mask, curr_time, stable_thresh,
                                   search_size, min_z_diff);

    Launch2DKernelCUDA(kernel, model.get_width(), model.get_height());
  } else {
    CarveSpaceKernel<kCPU> kernel(model, free_mask, curr_time, stable_thresh,
                                  search_size, min_z_diff);

    Launch2DKernelCPU(kernel, model.get_width(), model.get_height());
  }
}
}  // namespace fiontb