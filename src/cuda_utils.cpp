#include "cuda_utils.hpp"

namespace fiontb {
CudaKernelDims Get2DKernelDims(int width, int height) {
  dim3 block_dim = dim3(16, 16);
  dim3 grid_size(width / block_dim.x + 1, height / block_dim.y + 1);
  return CudaKernelDims(grid_size, block_dim);
}
}  // namespace fiontb
