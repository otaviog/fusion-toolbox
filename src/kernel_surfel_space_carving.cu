#include <torch/torch.h>

#include "cuda_utils.hpp"
#include "error.hpp"
#include "math.hpp"

namespace fiontb {

namespace {

template <typename Float3Accessor, typename IndexAccessor>
struct Framebuffer {
  Framebuffer(const Float3Accessor position, const IndexAccessor index)
      : position(position), index(index) {}

  __device__ __host__ int width() const { return position.size(1); }
  __device__ __host__ int height() const { return position.size(0); }
  __device__ __host__ bool empty(int row, int col) const {
    return index[row][col][1] == 0;
  }

  inline __device__ __host__ int time(int row, int col) const {
    return index[row][col][2];
  }

  inline __device__ __host__ float confidence(int row, int col) const {
    return position[row][col][3];
  }

  const Float3Accessor position;
  const IndexAccessor index;
};

typedef Framebuffer<PackedAccessor<float, 3>, PackedAccessor<int32_t, 3>>
    CUDAFramebuffer;

typedef Framebuffer<torch::TensorAccessor<float, 3>,
                    torch::TensorAccessor<int32_t, 3>>
    CPUFramebuffer;

const int MAX_VIOLANTIONS = 1;

__global__ void CarveSpace_gpu_kernel(CUDAFramebuffer model,
                                      PackedAccessor<uint8_t, 1> free_mask,
                                      int curr_time, float stable_thresh,
                                      int search_size, float min_z_diff) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= model.height() || col >= model.width()) return;
  if (model.empty(row, col)) return;
  if (model.time(row, col) == curr_time &&
      model.confidence(row, col) >= stable_thresh)
    return;
  const float model_z = model.position[row][col][2];
  const int model_idx = model.index[row][col][0];

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
      const float stable_z = model.position[krow][kcol][2];
      if (stable_z - model_z > min_z_diff) {
        ++violantion_count;
      }
    }
  }

  if (violantion_count >= MAX_VIOLANTIONS) {
    free_mask[model_idx] = 1;
  }
}

void CarveSpace_cpu_kernel(CPUFramebuffer model,
                           torch::TensorAccessor<uint8_t, 1> free_mask,
                           int curr_time, float stable_thresh, int search_size,
                           float min_z_diff) {
  for (int row = 0; row < model.height(); ++row) {
    for (int col = 0; col < model.width(); ++col) {
      if (model.empty(row, col)) continue;
      if (model.time(row, col) == curr_time &&
          model.confidence(row, col) >= stable_thresh)
        continue;
      const float model_z = model.position[row][col][2];
      const int model_idx = model.index[row][col][0];

      int violantion_count = 0;

      const int start_row = max((row - search_size), 0);
      const int end_row = min((row + search_size), model.height() - 1);

      const int start_col = max((col - search_size), 0);
      const int end_col = min((col + search_size), model.width() - 1);

      for (int krow = start_row; krow <= end_row; ++krow) {
        for (int kcol = start_col; kcol <= end_col; ++kcol) {
          if (krow == row && kcol == col) continue;
          if (model.empty(krow, kcol)) continue;

          if (model.time(krow, kcol) != curr_time &&
              model.confidence(krow, kcol) < stable_thresh)
            continue;

          const float stable_z = model.position[krow][kcol][2];

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

void CarveSpace(const torch::Tensor model_pos_fb,
                const torch::Tensor model_idx_fb, torch::Tensor free_mask,
                int curr_time, float stable_thresh, int search_size,
                float min_z_diff) {
  FTB_CHECK(free_mask.is_cuda(), "Expected a cuda tensor");

  if (model_pos_fb.is_cuda()) {
    FTB_CHECK(model_idx_fb.is_cuda(), "Expected a cuda tensor");

    CUDAFramebuffer model(
        model_pos_fb
            .packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>(),
        model_idx_fb
            .packed_accessor<int32_t, 3, torch::RestrictPtrTraits, size_t>());

    const CudaKernelDims kern_dims =
        Get2DKernelDims(model.width(), model.height());
    CarveSpace_gpu_kernel<<<kern_dims.grid, kern_dims.block>>>(
        model,
        free_mask
            .packed_accessor<uint8_t, 1, torch::RestrictPtrTraits, size_t>(),
        curr_time, stable_thresh,
        search_size, min_z_diff);
    CudaCheck();
    CudaSafeCall(cudaDeviceSynchronize());
  } else {
    FTB_CHECK(!model_idx_fb.is_cuda(), "Expected a CPU tensor");

    CPUFramebuffer model(model_pos_fb.accessor<float, 3>(),
                         model_idx_fb.accessor<int32_t, 3>());

    torch::Tensor free_mask_cpu = free_mask.cpu();
    CarveSpace_cpu_kernel(model, free_mask_cpu.accessor<uint8_t, 1>(),
                          curr_time, stable_thresh, search_size, min_z_diff);
    free_mask.copy_(free_mask_cpu);
  }
}
}  // namespace fiontb