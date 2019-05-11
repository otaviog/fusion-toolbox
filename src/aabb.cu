#include <iostream>

#include <torch/torch.h>

#include "cuda_error.hpp"

__global__ void KernIsInsideIndexed(const torch::PackedTensorAccessor<long, 1, torch::RestrictPtrTraits, size_t>
									indices,
									const torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits,
									size_t>
									points,
									float min_x, float min_y, float min_z,
									float max_x, float max_y, float max_z,
									torch::PackedTensorAccessor<uint8_t, 1, torch::RestrictPtrTraits, size_t>
									mask) {
  const long idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < indices.size(0)) {
	const long point_idx = indices[idx];
	mask[idx] = 0;
	
	if ((points[point_idx][0] >= min_x && points[point_idx][0] <= max_x)
		&& (points[point_idx][1] >= min_y && points[point_idx][1] <= max_y)
		&& (points[point_idx][2] >= min_z && points[point_idx][2] <= max_z)) {
	  mask[idx] = 1;
	}

  }
}

__global__ void KernIsInside(const torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits,
							 size_t> points,
							 float min_x, float min_y, float min_z, float max_x, float max_y,
							 float max_z,
							 torch::PackedTensorAccessor<uint8_t, 1, torch::RestrictPtrTraits, size_t>
							 mask) {
  
  const long idx = blockIdx.x;
  if (idx < points.size(0)) {
	mask[idx] = 0;
	
	if ((points[idx][0] >= min_x && points[idx][0] <= max_x)
		&& (points[idx][1] >= min_y && points[idx][1] <= max_y)
		&& (points[idx][2] >= min_z && points[idx][2] <= max_z)) {
	  mask[idx] = 1;
	}
  }
}

__global__ void KernIsInsideRadius(
							 const torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits,
							 size_t>
							 points, float radius,
							 float min_x, float min_y, float min_z,
							 float max_x, float max_y, float max_z,
							 torch::PackedTensorAccessor<uint8_t, 1, torch::RestrictPtrTraits, size_t>
							 mask) {
  const long idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < points.size(0)) {
	mask[idx] = 0;
	
	float point_x = points[idx][0];
	float point_y = points[idx][1];
	float point_z = points[idx][2];
	
	float closest_x = max(min(point_x, max_x), min_x);
	float closest_y = max(min(point_y, max_y), min_y);
	float closest_z = max(min(point_z, max_z), min_z);

	const float dist_x = (point_x - closest_x);
	const float dist_y = (point_y - closest_y);
	const float dist_z = (point_z - closest_z);

	const float dist = dist_x*dist_x + dist_y*dist_y + dist_z*dist_z;

	if (dist <= radius*radius) {
	  mask[idx] = 1;
	}
  }
}

__global__ void KernIsInsideRadius(const torch::PackedTensorAccessor<long, 1, torch::RestrictPtrTraits, size_t>
									indices,
							 const torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits,
							 size_t>
							 points, float radius,
							 float min_x, float min_y, float min_z,
							 float max_x, float max_y, float max_z,
							 torch::PackedTensorAccessor<uint8_t, 1, torch::RestrictPtrTraits, size_t>
							 mask) {
  const long idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < indices.size(0)) {
	const long point_idx = indices[idx];
	
	mask[idx] = 0;
	
	float point_x = points[point_idx][0];
	float point_y = points[point_idx][1];
	float point_z = points[point_idx][2];
	
	float closest_x = max(min(point_x, max_x), min_x);
	float closest_y = max(min(point_y, max_y), min_y);
	float closest_z = max(min(point_z, max_z), min_z);

	const float dist_x = (point_x - closest_x);
	const float dist_y = (point_y - closest_y);
	const float dist_z = (point_z - closest_z);

	const float dist = dist_x*dist_x + dist_y*dist_y + dist_z*dist_z;

	if (dist <= radius*radius) {
	  mask[idx] = 1;
	}
  }
}

namespace fiontb {
  torch::Tensor GPUIsInside(const torch::Tensor &indices,
							 const torch::Tensor &points, float min[3],
							 float max[3]) {
	torch::TensorOptions opts;
	opts = opts.dtype(torch::kUInt8).device(points.device());

	torch::Tensor mask = torch::empty({indices.size(0)}, opts);

	auto idx_acc =
	  indices.packed_accessor<long, 1, torch::RestrictPtrTraits, size_t>();
	auto pt_acc =
	  points.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>();
	auto mk_acc = mask.packed_accessor<uint8_t, 1, torch::RestrictPtrTraits,
	  size_t>();

	
	KernIsInsideIndexed<<<(indices.size(0)/1024)+1, 1024>>>(idx_acc, pt_acc, min[0], min[1], min[2],
												max[0], max[1], max[2], mk_acc);
	CudaCheck();
	return mask;
  }

  torch::Tensor GPUIsInside(const torch::Tensor &points, float min[3],
                             float max[3]) {
    torch::TensorOptions opts;
    opts = opts.dtype(torch::kUInt8).device(points.device());

    torch::Tensor mask = torch::empty({points.size(0)}, opts);

    auto pt_acc =
	  points.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>();
    auto mk_acc =
	  mask.packed_accessor<uint8_t, 1, torch::RestrictPtrTraits, size_t>();

    KernIsInside<<<points.size(0), 1>>>(pt_acc, min[0], min[1], min[2], max[0],
                                        max[1], max[2], mk_acc);
	CudaCheck();
    return mask;
  }

  torch::Tensor GPUIsInside(const torch::Tensor &points, float radius,
							 float min[3], float max[3]) {
    torch::TensorOptions opts;
    opts = opts.dtype(torch::kUInt8).device(points.device());

	const int num_points = points.size(0);
    torch::Tensor mask = torch::full({num_points}, 2, opts);

    auto pt_acc =
	  points.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>();
    auto mk_acc =
	  mask.packed_accessor<uint8_t, 1, torch::RestrictPtrTraits, size_t>();

    KernIsInsideRadius<<<(num_points/1024)+1, 1024>>>(pt_acc, radius, min[0], min[1], min[2], max[0],
													  max[1], max[2], mk_acc);
	CudaCheck();
    return mask;
  }

  torch::Tensor GPUIsInside(const torch::Tensor &indices,
							 const torch::Tensor &points, float radius,
							 float min[3], float max[3]) {
    torch::TensorOptions opts;
    opts = opts.dtype(torch::kUInt8).device(points.device());

	const int num_indices = indices.size(0);
    torch::Tensor mask = torch::full({num_indices}, 2, opts);

	auto idx_acc = indices.packed_accessor<long, 1, torch::RestrictPtrTraits, size_t>();
	
    auto pt_acc =
	  points.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>();
    auto mk_acc =
	  mask.packed_accessor<uint8_t, 1, torch::RestrictPtrTraits, size_t>();

    KernIsInsideRadius<<<(num_indices/1024)+1, 1024>>>(idx_acc, pt_acc, radius, min[0], min[1], min[2], max[0],
													  max[1], max[2], mk_acc);
	CudaCheck();
    return mask;
  }

}