#pragma once

#include "pointgrid.hpp"

namespace fiontb {
namespace {
template <Device dev>
class PointGrid : public BasePointGrid<dev> {
 public:
  PointGrid(const torch::Tensor &points, const torch::Tensor &mask)
      : BasePointGrid<dev>(mask),
        points(Accessor<dev, float, 3>::Get(points)) {}

  typename Accessor<dev, float, 3>::T points;
};

}  // namespace

};  // namespace fiontb
