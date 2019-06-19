#include <torch/torch.h>

#include "cuda_error.hpp"

namespace fiontb {
  __global__ void KernCarveSpace(
								 const torch::PackedTensorAccessor<float, 3, torch::RestrictPtrTraits,
								 size_t>
								 stable_pos_fb,
								 const torch::PackedTensorAccessor<int, 2, torch::RestrictPtrTraits, size_t>
								 stable_idx_fb,
								 const torch::PackedTensorAccessor<float, 3, torch::RestrictPtrTraits,
								 size_t>
								 view_pos_fb,
								 const torch::PackedTensorAccessor<int, 2, torch::RestrictPtrTraits, size_t>
								 view_idx_fb,
								 torch::PackedTensorAccessor<uint8_t, 1, torch::RestrictPtrTraits, size_t>
								 mask, int neighbor_size) {
	const int row = blockIdx.y * blockDim.y + threadIdx.y;
	const int col = blockIdx.x * blockDim.x + threadIdx.x;

	const int width = view_pos_fb.size(1);
	const int height = view_pos_fb.size(0);
	if (row >= height || col >= width) {
	  return;
	}

	int violantion_count = 0;

	const float view_pos_z = view_pos_fb[row][col][2];
	for (int krow = max(0, row - neighbor_size);
		 krow < min(height - 1, row + neighbor_size); ++krow) {
	  for (int kcol = max(0, col - neighbor_size);
		   kcol < min(width - 1, col + neighbor_size); ++kcol) {

		const int stable_idx = stable_idx_fb[krow][kcol] - 1;
		if (stable_idx == 0) {
		  continue;
		}
		
		const float stable_z = stable_pos_fb[krow][kcol][2];
		if ((stable_z - view_pos_z) > 0.1) {
		  ++violantion_count;
		}
	  }
	}

	if (violantion_count > 1) {
	  const int view_idx = view_idx_fb[row][col] - 1;
	  mask[view_idx] = 1;
	}
  }

  void CarveSpace(const torch::Tensor stable_pos_fb,
				  const torch::Tensor stable_idx_fb,
				  const torch::Tensor view_pos_fb,
				  const torch::Tensor view_idx_fb,
				  torch::Tensor mask, int neighbor_size) {
	const int width = view_pos_fb.size(1);
	const int height = view_pos_fb.size(0);
	
	dim3 block_dim = dim3(16, 16);
	
	dim3 grid_size(width / block_dim.x +  1,
				   height / block_dim.y + 1);
        KernCarveSpace<<<grid_size, block_dim>>>(
            stable_pos_fb
                .packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>(),
			stable_idx_fb.packed_accessor<int32_t, 2, torch::RestrictPtrTraits,
                                        size_t>(),
            view_pos_fb
                .packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>(),
            view_idx_fb.packed_accessor<int32_t, 2, torch::RestrictPtrTraits,
                                        size_t>(),
            mask.packed_accessor<uint8_t, 1, torch::RestrictPtrTraits,
                                 size_t>(),
            neighbor_size);
        CudaCheck();
  }
}  // namespace fiontb

