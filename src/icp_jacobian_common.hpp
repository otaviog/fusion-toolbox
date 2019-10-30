#pragma once

#include "feature_map.hpp"
#include "pointgrid.hpp"

namespace fiontb {

namespace {
template <Device dev, typename scalar_t>
class PointGrid : public BasePointGrid<dev> {
 public:
  const typename Accessor<dev, scalar_t, 3>::T points;
  const typename Accessor<dev, scalar_t, 3>::T normals;

  PointGrid(const torch::Tensor &points, const torch::Tensor normals,
            const torch::Tensor &mask)
      : BasePointGrid<dev>(mask),
        points(Accessor<dev, scalar_t, 3>::Get(points)),
        normals(Accessor<dev, scalar_t, 3>::Get(normals)) {}
};

template <Device dev, typename scalar_t>
FTB_DEVICE_HOST inline scalar_t EuclideanDistance(
    const BilinearInterp<dev, scalar_t> f1,
    const typename Accessor<dev, scalar_t, 2>::T f2, int f2_index) {
  scalar_t dist = scalar_t(0);
  for (int channel = 0; channel < f2.size(0); ++channel) {
    const scalar_t diff = f1.Get(channel) - f2[channel][f2_index];
    dist += diff * diff;
  }

  return sqrt(dist);
}

template <typename scalar_t>
FTB_DEVICE_HOST inline scalar_t Df1_EuclideanDistance(
    scalar_t f1_nth_val, scalar_t f2_nth_val, scalar_t inv_forward_result) {
  if (inv_forward_result > 0)
    return (f1_nth_val - f2_nth_val) * inv_forward_result;
  else
    return 0;
}
}  // namespace
}  // namespace fiontb
