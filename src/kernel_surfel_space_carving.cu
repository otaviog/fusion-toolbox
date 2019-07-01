#include <torch/torch.h>

#include "cuda_utils.hpp"
#include "math.hpp"

namespace fiontb {

namespace {
__global__ void CarveSpace_gpu_kernel(
    const PackedAccessor<float, 3> stable_pos_fb,
    const PackedAccessor<int32_t, 3> stable_idx_fb,
    const PackedAccessor<float, 3> view_pos_fb,
    const PackedAccessor<int32_t, 3> view_idx_fb,
    PackedAccessor<uint8_t, 1> free_mask, int neighbor_size) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  const int width = view_pos_fb.size(1);
  const int height = view_pos_fb.size(0);

  if (row >= height || col >= width) return;
  if (view_idx_fb[row][col][1] == 0) return;

  int violantion_count = 0;

  const float view_pos_z = view_pos_fb[row][col][2];
  for (int krow = max(0, row - neighbor_size);
       krow < min(height - 1, row + neighbor_size); ++krow) {
    for (int kcol = max(0, col - neighbor_size);
         kcol < min(width - 1, col + neighbor_size); ++kcol) {
      const int stable_flag = stable_idx_fb[krow][kcol][1];

      if (stable_flag == 0) {
        continue;
      }

      const int stable_idx = stable_idx_fb[krow][kcol][0];
      const float stable_z = stable_pos_fb[krow][kcol][2];
      if ((stable_z - view_pos_z) > 0.1) {
        ++violantion_count;
      }
    }
  }

  if (violantion_count > 1) {
    const int view_idx = view_idx_fb[row][col][0];
    free_mask[view_idx] = 1;
  }
}
}  // namespace

void CarveSpace(const torch::Tensor stable_pos_fb,
                const torch::Tensor stable_idx_fb,
                const torch::Tensor view_pos_fb,
                const torch::Tensor view_idx_fb, torch::Tensor free_mask,
                int neighbor_size) {
  const int width = view_pos_fb.size(1);
  const int height = view_pos_fb.size(0);

  const CudaKernelDims kern_dims = Get2DKernelDims(width, height);

  CarveSpace_gpu_kernel<<<kern_dims.grid, kern_dims.block>>>(
      stable_pos_fb
          .packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>(),
      stable_idx_fb
          .packed_accessor<int32_t, 3, torch::RestrictPtrTraits, size_t>(),
      view_pos_fb.packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>(),
      view_idx_fb
          .packed_accessor<int32_t, 3, torch::RestrictPtrTraits, size_t>(),
      free_mask.packed_accessor<uint8_t, 1, torch::RestrictPtrTraits, size_t>(),
      neighbor_size);
  CudaCheck();
}
}  // namespace fiontb