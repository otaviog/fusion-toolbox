#pragma once

#include <torch/torch.h>

namespace pybind11 {
class module;
}

namespace slamtb {
/**
 * Functions to calculate the ICP jacobians w.r.t. to the Lie algebra
 * transformation parameters.
 */
struct ICPJacobian {
  /**
   * Computes the \f$\mathfrak{se}(3)\f$ jacobian of the
   * point-to-plane ICP error function.
   *
   * This function does the following:
   *
   * - Computes the \f$ \sum_{i, m}^{FindClosestPoints} (tgt\_points_m
   * - src\_points_i)\cdot dst\_normals_m\f$ error.
   *
   * - As it is designed to work with Gauss Newton, it returns the per
   *   source point \f$ J^\top J\f$ w.r.t. the 6
   *   \f$\mathfrak{se}(3)\f$ parameters.
   *
   * - Returns the the per point \f$ Jr \f$.
   *
   * - Correspondences are filtered based on distance and normal
   *   angles thresholds.
   *
   * @param tgt_points Target's cloud points. Float or double [HxWx3]
   * tensor.
   *
   * @param tgt_normals Target's cloud normals. Float or double
   * [HxWx3] tensor.
   *
   * @param tgt_mask Valid target entries. Bool [HxW] tensor.
   *
   * @param src_points Source's cloud points. Float or double [Nx3]
   * tensor.
   *
   * @param src_normals Source's cloud normals. Float or double [Nx3]
   * tensor.
   *
   * @param src_mask Valid source entries. Bool [N] tensor.
   *
   * @param kcam Camera intrinsics matrix. Float or double [3x3] tensor.
   *
   * @param rt_cam Rigid transformation matrix. Float or double is [3x4] or
   * [4x4] tensor.
   *
   * @param distance_thresh Points with distance higher than this
   * are set as non-matches.
   *
   * @param normals_angle_thresh Maximum angle in radians between
   * normals to match a pair of source and target points.
   *
   * @param JtJ_partial Transposed Jacobian @ Jacobian partial
   * matrices. To have the final (\f$ J^\top J\f$), sum all N
   * matrices. Same as type as inputs [Nx6x6] tensor.
   *
   * @param Jr_partial Jacobian @ resisual (\f$Jr\f$). Sum all N
   * vectors for the final value. Same type as inputs [Nx6] tensor.
   *
   * @param squared_residual The squared residual value of the cost
   * function. Same type as input [N] tensor.
   *
   * @return The number of matches.
   */
  static int EstimateGeometric(
      const torch::Tensor &tgt_points, const torch::Tensor &tgt_normals,
      const torch::Tensor &tgt_mask, const torch::Tensor &src_points,
      const torch::Tensor &src_normals, const torch::Tensor &src_mask,
      const torch::Tensor &kcam, const torch::Tensor &rt_cam,
      float distance_thresh, float normals_angle_thresh,
      torch::Tensor JtJ_partial, torch::Tensor Jr_partial,
      torch::Tensor squared_residual);

  /**
   * Computes the \f$\mathfrak{se}(3)\f$ jacobian of the feature
   * reprojection error.
   *
   * This function does the following:
   *
   * - Computes the \f$ \sum_{i} ||tgt\_feat[\pi(rt\_cam
       src\_points_i)] - src\_feat[\pi(src\_points_i)]||_2 \f$ error.
   *
   * - As it is designed to work with Gauss Newton, it returns the per
   *   source point \f$ J^\top J\f$ w.r.t. the 6 parameters of
   *   \f$\mathfrak{se}(3)\f$.
   *
   * - Returns the the per point \f$ Jr \f$.
   *
   * - Correspondences are filtered based on distance and normal
   *   angles thresholds.
   *
   * @param tgt_points Target's cloud points. Float or double [HxWx3]
   * tensor.
   *
   * @param tgt_normals Target's cloud normals. Float or double
   * [HxWx3] tensor.

   * @param tgt_feats Target's feature map. Float or double [FxHxW] tensor.
   *
   * @param tgt_mask Valid target entries. Bool [HxW] tensor.
   *
   * @param src_points Source's cloud points. Float or double [Nx3]
   * tensor.
   *
   * @param src_feats Source's feature map. Float or double [FxHxW]
   * tensor.
   *
   * @param src_mask Valid source entries. Bool [N] tensor.
   *
   * @param kcam Camera intrinsics matrix. Float or double [3x3] tensor.
   *
   * @param rt_cam rigid transformation matrix. Float or double is [3x4] or
   * [4x4] tensor.
   *
   * @param JtJ_partial Transposed Jacobian @ Jacobian partial
   * matrices. To have the final (\f$ J^\top J\f$), sum all N
   * matrices. Same as type as inputs [Nx6x6] tensor.
   *
   * @param Jr_partial Jacobian @ resisual (\f$Jr\f$). Sum all N
   * vectors for the final value. Same type as inputs [Nx6] tensor.
   *
   * @param squared_residual The squared residual value of the cost
   * function. Same type as input [N] tensor.
   *
   * @return The number of matches.
   */
  static int EstimateFeature(
      const torch::Tensor &src_points, const torch::Tensor &src_feats,
      const torch::Tensor &src_mask, const torch::Tensor &rt_cam,
      const torch::Tensor &tgt_feats, const torch::Tensor &kcam,
      const torch::Tensor merge_map, float residual_thresh,
      torch::Tensor JtJ_partial, torch::Tensor Jr_partial,
      torch::Tensor squared_residuals);

  /**
   * Computes the \f$\mathfrak{so}(3)\f$ jacobian of the feature
   * reprojection error.
   *
   * This function does the following:
   *
   * - Computes the \f$ \sum_{i} ||tgt\_feat[\pi(rt\_cam
       src\_points_i)] - src\_feat[\pi(src\_points_i)]||_2 \f$ error.
   *
   * - As it is designed to work with Gauss Newton, it returns the per
   *   source point \f$ J^\top J\f$ w.r.t. the 6 parameters of
   *   \f$\mathfrak{se}(3)\f$.
   *
   * - Returns the the per point \f$ Jr \f$.
   *
   * - Correspondences are filtered based on distance and normal
   *   angles thresholds.
   *
   * @param tgt_points Target's cloud points. Float or double [HxWx3]
   * tensor.
   *
   * @param tgt_normals Target's cloud normals. Float or double
   * [HxWx3] tensor.

   * @param tgt_feats Target's feature map. Float or double [FxHxW] tensor.
   *
   * @param tgt_mask Valid target entries. Bool [HxW] tensor.
   *
   * @param src_points Source's cloud points. Float or double [Nx3]
   * tensor.
   *
   * @param src_feats Source's feature map. Float or double [FxHxW]
   * tensor.
   *
   * @param src_mask Valid source entries. Bool [N] tensor.
   *
   * @param kcam Camera intrinsics matrix. Float or double [3x3] tensor.
   *
   * @param rt_cam rigid transformation matrix. Float or double is [3x4] or
   * [4x4] tensor.
   *
   * @param JtJ_partial Transposed Jacobian @ Jacobian partial
   * matrices. To have the final (\f$ J^\top J\f$), sum all N
   * matrices. Same as type as inputs [Nx6x6] tensor.
   *
   * @param Jr_partial Jacobian @ resisual (\f$Jr\f$). Sum all N
   * vectors for the final value. Same type as inputs [Nx6] tensor.
   *
   * @param squared_residual The squared residual value of the cost
   * function. Same type as input [N] tensor.
   *
   * @return The number of matches.
   */
  static int EstimateFeatureSO3(
      const torch::Tensor &src_points, const torch::Tensor &src_feats,
      const torch::Tensor &src_mask, const torch::Tensor &rt_cam,
      const torch::Tensor &tgt_feats, const torch::Tensor &kcam,
      const torch::Tensor merge_map, float residual_thresh,
      torch::Tensor JtJ_partial, torch::Tensor Jr_partial,
      torch::Tensor squared_residuals);

  /**
   * Register it in Pybind.
   */
  static void RegisterPybind(pybind11::module &m);
};
}  // namespace slamtb
