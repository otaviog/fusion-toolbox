#include <torch/torch.h>

#include "cuda_utils.hpp"
#include "error.hpp"
#include "math.hpp"

namespace fiontb {

namespace {

template <typename PositionAccessor,
  typename IndexAccessor>
struct Framebuffer {
  Framebuffer(const PositionAccessor position, const IndexAccessor index)
      : position(position), index(index) {}

  __device__ __host__ int width() const { return position.size(1); }
  __device__ __host__ int height() const { return position.size(0); }
  __device__ __host__ bool empty(int row, int col) const {
    return index[row][col][1] == 0;
  }

  const PositionAccessor position;
  const IndexAccessor index;
};

typedef Framebuffer<PackedAccessor<float, 3>, PackedAccessor<int32_t, 3>>
    CUDAFramebuffer;

typedef Framebuffer<torch::TensorAccessor<float, 3>,
                    torch::TensorAccessor<int32_t, 3>>
    CPUFramebuffer;

const int MAX_VIOLANTIONS = 4;

__global__ void CarveSpace_gpu_kernel(CUDAFramebuffer stable_and_new,
                                      CUDAFramebuffer model,
                                      PackedAccessor<uint8_t, 1> free_mask,
                                      int search_size, float min_z_diff) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= model.height() || col >= model.width()) return;
  if (model.empty(row, col)) return;

  const float model_z = model.position[row][col][2];
  const int model_idx = model.index[row][col][0];

  // if (model.position[row][col][3] < 20) return;

  /**
   * Stable_And_New and model position framebuffers may differ in size.
   */
  const int wscale = model.width() / stable_and_new.width();
  const int hscale = model.height() / stable_and_new.height();

  int violantion_count = 0;

  const int start_row = max((row - search_size) / hscale, 0);
  const int end_row = min((row + search_size) / hscale, model.height() - 1);

  const int start_col = max((col - search_size) / wscale, 0);
  const int end_col = min((col + search_size) / wscale, model.width() - 1);

  for (int krow = start_row; krow <= end_row; ++krow) {
    for (int kcol = start_col; kcol <= end_col; ++kcol) {
      if (stable_and_new.empty(krow, kcol)) continue;
      const float stable_z = stable_and_new.position[krow][kcol][2];

      if (stable_z - model_z > min_z_diff) {
        ++violantion_count;
      }
    }
  }

  if (violantion_count >= MAX_VIOLANTIONS) {
    free_mask[model_idx] = 1;
  }
}

void CarveSpace_cpu_kernel(CPUFramebuffer stable_and_new, CPUFramebuffer model,
                           torch::TensorAccessor<uint8_t, 1> free_mask,
                           int search_size, float min_z_diff) {
  /**
   * Stable_And_New and model position framebuffers may differ in size.
   */
  const int wscale = model.width() / stable_and_new.width();
  const int hscale = model.height() / stable_and_new.height();

  for (int row = 0; row < model.height(); ++row) {
    for (int col = 0; col < model.width(); ++col) {
      if (model.empty(row, col)) continue;
      const float model_z = model.position[row][col][2];

      const int model_idx = model.index[row][col][0];

      int violantion_count = 0;

      const int start_row = max((row - search_size) / hscale, 0);
      const int end_row = min((row + search_size) / hscale, model.height() - 1);

      const int start_col = max((col - search_size) / wscale, 0);
      const int end_col = min((col + search_size) / wscale, model.width() - 1);

      for (int krow = start_row; krow <= end_row; ++krow) {
        for (int kcol = start_col; kcol <= end_col; ++kcol) {
          if (stable_and_new.empty(krow, kcol)) continue;
          const float stable_z = stable_and_new.position[krow][kcol][2];
          
          if (stable_z - model_z > min_z_diff) {
            ++violantion_count;
          }
        }
      }

      if (violantion_count >= MAX_VIOLANTIONS) {
        free_mask[model_idx] = 1;
      }
    }
  }
}

}  // namespace

void CarveSpace(const torch::Tensor stable_and_new_pos_fb,
                const torch::Tensor stable_and_new_idx_fb,
                const torch::Tensor model_pos_fb,
                const torch::Tensor model_idx_fb, torch::Tensor free_mask,
                int search_size, float min_z_diff) {

  FTB_CHECK(free_mask.is_cuda(), "Expected a cuda tensor");
  
  if (stable_and_new_pos_fb.is_cuda()) {
    FTB_CHECK(stable_and_new_idx_fb.is_cuda(), "Expected a cuda tensor");
    FTB_CHECK(model_pos_fb.is_cuda(), "Expected a cuda tensor");
    FTB_CHECK(model_idx_fb.is_cuda(), "Expected a cuda tensor");
        
    CUDAFramebuffer stable_and_new(
        stable_and_new_pos_fb
            .packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>(),
        stable_and_new_idx_fb
            .packed_accessor<int32_t, 3, torch::RestrictPtrTraits, size_t>());

    CUDAFramebuffer model(
        model_pos_fb
            .packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>(),
        model_idx_fb
            .packed_accessor<int32_t, 3, torch::RestrictPtrTraits, size_t>());

    const CudaKernelDims kern_dims =
        Get2DKernelDims(model.width(), model.height());
    CarveSpace_gpu_kernel<<<kern_dims.grid, kern_dims.block>>>(
        stable_and_new, model,
        free_mask
            .packed_accessor<uint8_t, 1, torch::RestrictPtrTraits, size_t>(),
        search_size, min_z_diff);
    CudaCheck();
    CudaSafeCall(cudaDeviceSynchronize());
  } else {
    FTB_CHECK(!stable_and_new_idx_fb.is_cuda(), "Expected a CPU tensor");
    FTB_CHECK(!model_pos_fb.is_cuda(), "Expected a CPU tensor");
    FTB_CHECK(!model_idx_fb.is_cuda(), "Expected a CPU tensor");
    CPUFramebuffer stable_and_new(stable_and_new_pos_fb.accessor<float, 3>(),
                                  stable_and_new_idx_fb.accessor<int32_t, 3>());
    CPUFramebuffer model(model_pos_fb.accessor<float, 3>(),
                         model_idx_fb.accessor<int32_t, 3>());

    torch::Tensor free_mask_cpu = free_mask.cpu();
    CarveSpace_cpu_kernel(stable_and_new, model,
                          free_mask_cpu.accessor<uint8_t, 1>(), search_size,
                          min_z_diff);
    free_mask.copy_(free_mask_cpu);
  }
}
}  // namespace fiontb