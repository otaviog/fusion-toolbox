#include "cuda_utils.hpp"
#include "math.hpp"

using namespace std;

namespace fiontb {

class UnalignedTensorGuard {
 public:
  UnalignedTensorGuard(const torch::Tensor &tensor) {
    if (tensor.storage_offset() != 0) {
      stringstream stream;
      stream << "Unaligned tensor guard";
      throw std::runtime_error(stream.str());
    }
  }
};

__global__ void RasterIndexmap_gpu_kernel(
    const PackedAccessor<float, 2> points,
    const PackedAccessor<float, 2> pm,
    PackedAccessor<long, 2> indexmap, PackedAccessor<int, 2> depth_buffer,
    int depth_scale) {
  const long point_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (point_idx >= points.size(0)) return;

  const int width = indexmap.size(1);
  const int height = indexmap.size(0);

  const Eigen::Vector4f hpoint(points[point_idx][0], points[point_idx][1],
                               points[point_idx][2], 1.0f);

  // TODO: figure out how to use Eigen::Matrix4f here
  const float px = pm[0][0] * hpoint[0] + pm[0][1] * hpoint[1] + pm[0][2] * hpoint[2] + pm[0][3];
  const float py = pm[1][0] * hpoint[0] + pm[1][1] * hpoint[1] + pm[1][2] * hpoint[2] + pm[1][3];
  const float pz = pm[2][0] * hpoint[0] + pm[2][1] * hpoint[1] + pm[2][2] * hpoint[2] + pm[2][3];
  const float pw = pm[3][0] * hpoint[0] + pm[3][1] * hpoint[1] + pm[3][2] * hpoint[2] + pm[3][3];

  // TODO: is doing this flip right?
  const int frag_x = width - (px / pw + 1.0f) * 0.5f * width;
  const int frag_y = height - (py / pw + 1.0f) * 0.5f * height;

  if (frag_x < 0 || frag_x >= width || frag_y < 0 || frag_y >= height) return;

  const int frag_depth = abs(pz/pw * depth_scale);

  int *depth_addr = &depth_buffer[frag_y][frag_x];
  while (true) {
    int depth = *depth_addr;
    if (depth >= 0 && frag_depth >= depth*depth_scale) return;

    int old = atomicCAS(depth_addr, depth, frag_depth);
    if (old == depth) {
      break;
    }
  }

  while (true) {
    int z = *zaddr;
    int idx = *idxaddr;

    if (atomicMin(zaddr, myz) == z) {
      if (atomicCAS(idxaddr, idx, my_idx) == idx)
    }
  }
  
  // Depth passed

  indexmap[frag_y][frag_x] = point_idx;
}

void RasterIndexmap(const torch::Tensor points,
                    const torch::Tensor proj_matrix, torch::Tensor indexmap,
                    torch::Tensor depth_buffer) {
  UnalignedTensorGuard guard(proj_matrix);

  const CudaKernelDims kern_lc = Get1DKernelDims(points.size(0));
  const int depth_scale = 16000;
  RasterIndexmap_gpu_kernel<<<kern_lc.grid, kern_lc.block>>>(
      points.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
      proj_matrix.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
      indexmap.packed_accessor<long, 2, torch::RestrictPtrTraits, size_t>(),
      depth_buffer.packed_accessor<int, 2, torch::RestrictPtrTraits, size_t>(),
      depth_scale);
  CudaCheck();
  CudaSafeCall(cudaDeviceSynchronize());
}
}  // namespace fiontb