#include "pointgrid.hpp"

namespace fiontb {
namespace {

template <Device dev>
class PointGrid : public BasePointGrid<dev> {
 public:
  PointGrid(const torch::Tensor &points, const torch::Tensor normals,
                  const torch::Tensor &mask)
      : BasePointGrid<dev>(mask),
        points(Accessor<dev, float, 3>::Get(points)),
        normals(Accessor<dev, float, 3>::Get(normals)) {}

  typename Accessor<dev, float, 3>::T points;
  typename Accessor<dev, float, 3>::T normals;
};

template <Device dev>
class IntensityPointGrid : public PointGrid<dev> {
 public:
  IntensityPointGrid(const torch::Tensor &image,
                           const torch::Tensor &grad_image,
                           const torch::Tensor &points,
                           const torch::Tensor normals,
                           const torch::Tensor &mask)
      : PointGrid<dev>(points, normals, mask),
        image(Accessor<dev, float, 2>::Get(image)),
        grad_image(Accessor<dev, float, 3>::Get(grad_image)) {}

  typename Accessor<dev, float, 2>::T image;
  typename Accessor<dev, float, 3>::T grad_image;
};
}  // namespace

}  // namespace fiontb
