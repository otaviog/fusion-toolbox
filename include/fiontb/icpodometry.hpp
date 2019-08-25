#pragma once

#include <vector>

#include <torch/torch.h>

namespace fiontb {

void EstimateJacobian_cpu(const torch::Tensor points0,
                          const torch::Tensor normals0,
                          const torch::Tensor mask0,
                          const torch::Tensor points1,
                          const torch::Tensor mask1, const torch::Tensor kcam,
                          const torch::Tensor params, torch::Tensor jacobian,
                          torch::Tensor residual);

void EstimateJacobian_gpu(const torch::Tensor points0,
                          const torch::Tensor normals0,
                          const torch::Tensor mask0,
                          const torch::Tensor points1,
                          const torch::Tensor mask1, const torch::Tensor kcam,
                          const torch::Tensor params, torch::Tensor jacobian,
                          torch::Tensor residual);

void CalcSobelGradient_gpu(const torch::Tensor image, torch::Tensor out_grad);

void EstimateIntensityJacobian_gpu(
    const torch::Tensor tgt_points, const torch::Tensor tgt_normals,
    const torch::Tensor tgt_image, const torch::Tensor tgt_grad_image,
    const torch::Tensor tgt_mask, const torch::Tensor src_points,
    const torch::Tensor src_intensity, const torch::Tensor src_mask,
    const torch::Tensor kcam, const torch::Tensor rt_cam,
    torch::Tensor jacobian, torch::Tensor residual);

void EstimateIntensityJacobian_cpu(
    const torch::Tensor tgt_points, const torch::Tensor tgt_normals,
    const torch::Tensor tgt_image, const torch::Tensor tgt_grad_image,
    const torch::Tensor tgt_mask, const torch::Tensor src_points,
    const torch::Tensor src_intensity, const torch::Tensor src_mask,
    const torch::Tensor kcam, const torch::Tensor rt_cam,
    torch::Tensor jacobian, torch::Tensor residual);

}  // namespace fiontb
