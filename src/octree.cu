#include <torch/torch.h>

#include <cuda_runtime.h>

#include "cuda_error.hpp"

typedef torch::PackedTensorAccessor<float, 1, torch::RestrictPtrTraits,
														  size_t> PackedFloatAccessor1;

typedef torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits,
														  size_t> PackedFloatAccessor2;

typedef torch::PackedTensorAccessor<long, 1, torch::RestrictPtrTraits,
														  size_t> PackedLongAccessor1;

typedef torch::PackedTensorAccessor<long, 2, torch::RestrictPtrTraits,
														  size_t> PackedLongAccessor2;

__global__ void KernDistances(const PackedLongAccessor1 qpoints_idxs,
							  const PackedFloatAccessor2 qpoints,
							  const PackedLongAccessor1 mpoints_idxs,
							  const PackedFloatAccessor2 mpoints,
							  PackedFloatAccessor2 distances) {
  
  const long qidx = blockIdx.x*blockDim.x + threadIdx.x;
  const long midx = blockIdx.y*blockDim.y + threadIdx.y;
  
  if (qidx < qpoints_idxs.size(0) && midx < mpoints_idxs.size(0)) {
	const long qpoint_idx = qpoints_idxs[qidx];
	const long mpoint_idx = mpoints_idxs[midx];

	const float qx = qpoints[qpoint_idx][0];
	const float qy = qpoints[qpoint_idx][1];
	const float qz = qpoints[qpoint_idx][2];

	const float mx = mpoints[mpoint_idx][0];
	const float my = mpoints[mpoint_idx][1];
	const float mz = mpoints[mpoint_idx][2];

	float x = mx - qx;
	float y = my - qy;
	float z = mz - qz;
	
	distances[qidx][midx] = x*x + y*y + z*z;
  }

}

__global__ void KernDistances(long qidx, float qx, float qy, float qz,
							  const PackedLongAccessor1 mpoints_idxs,
							  const PackedFloatAccessor2 mpoints,
							  PackedFloatAccessor2 distances) {
  const long midx = blockIdx.x*blockDim.x + threadIdx.x;
  
  if (midx < mpoints_idxs.size(0)) {
	const long mpoint_idx = mpoints_idxs[midx];
	
	const float mx = mpoints[mpoint_idx][0];
	const float my = mpoints[mpoint_idx][1];
	const float mz = mpoints[mpoint_idx][2];

	float x = mx - qx;
	float y = my - qy;
	float z = mz - qz;
	
	distances[0][midx] = x*x + y*y + z*z;
  }

}

__global__ void KernCopyResultToTensor(const PackedFloatAccessor2 sorted_dists,
									   const PackedLongAccessor2 sorted_indices,
									   const PackedLongAccessor1 indices,
									   int qidx, float radius,
									   PackedFloatAccessor2 out_distances,
									   PackedLongAccessor2 out_indices) {
  const long idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < sorted_dists.size(1)) {	
	const float dist = sorted_dists[0][idx];
	if (dist < radius*radius) {
	  //if (true) {
	  out_distances[qidx][idx] = sqrt(dist);
	  out_indices[qidx][idx] = indices[sorted_indices[0][idx]];
	}
  }
}

namespace fiontb {
  torch::Tensor GPUPointDistances(const torch::Tensor &qpoints_idxs,
							   const torch::Tensor &qpoints,
							   const torch::Tensor &mpoints_idxs,
							   const torch::Tensor &mpoints) {
	const int num_qpoints = qpoints_idxs.size(0);
	const int num_mpoints = mpoints_idxs.size(0);
	
	dim3 block_dim(16, 16, 1);
	dim3 grid_dim(num_mpoints/block_dim.x + 1, num_mpoints/block_dim.y + 1, 1);

	torch::TensorOptions opts;
	opts = opts.dtype(torch::kFloat).device(qpoints.device());
	
	torch::Tensor distances = torch::empty({num_qpoints, num_mpoints}, opts);
	KernDistances<<<grid_dim, block_dim>>>(qpoints_idxs.packed_accessor<long, 1, torch::RestrictPtrTraits, size_t>(),
										   qpoints.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
										   mpoints_idxs.packed_accessor<long, 1, torch::RestrictPtrTraits, size_t>(),
										   mpoints.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
										   distances.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>());

	CudaCheck();
	return distances;
  }

  torch::Tensor GPUPointDistances(int qidx,
							   float qx, float qy, float qz,
							   const torch::Tensor &mpoints_idxs,
							   const torch::Tensor &mpoints) {
	const int num_mpoints = mpoints_idxs.size(0);
	
	torch::TensorOptions opts;
	opts = opts.dtype(torch::kFloat).device(mpoints.device());	
	torch::Tensor distances = torch::empty({1, num_mpoints}, opts);
	
	KernDistances<<<num_mpoints/1024 + 1, 1024>>>(qidx, qx, qy, qz,
										   mpoints_idxs.packed_accessor<long, 1, torch::RestrictPtrTraits, size_t>(),
										   mpoints.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
										   distances.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>());

	CudaCheck();
	return distances;
  }

  void GPUCopyResultToTensor(const torch::Tensor &sorted_dists,
							 const torch::Tensor &sorted_indices,
							 const torch::Tensor &indices,
							 int qidx, float radius,
							 torch::Tensor out_distances,
							 torch::Tensor out_indices) {
	const PackedFloatAccessor2 sorted_dists_acc = sorted_dists.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>();

	const PackedLongAccessor2 sorted_indices_acc = sorted_indices.packed_accessor<long, 2, torch::RestrictPtrTraits, size_t>();

	const PackedLongAccessor1 indices_acc = indices.packed_accessor<long, 1, torch::RestrictPtrTraits, size_t>();

	PackedFloatAccessor2 out_dist_acc = out_distances.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>();

	PackedLongAccessor2 out_dist_idx = out_indices.packed_accessor<long, 2, torch::RestrictPtrTraits, size_t>();

	const int num_dists = sorted_dists.size(1);
	KernCopyResultToTensor<<<1, num_dists>>>(sorted_dists_acc, sorted_indices_acc,
											 indices_acc, qidx, radius,
											 out_dist_acc, out_dist_idx);
	  CudaCheck();
  }
}