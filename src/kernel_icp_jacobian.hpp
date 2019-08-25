#include "pointgrid.hpp"

namespace fiontb {
namespace {

template <bool CUDA>
class PointGrid : public BasePointGrid<CUDA> {
 public:
  PointGrid(const torch::Tensor &points, const torch::Tensor normals,
                  const torch::Tensor &mask)
      : BasePointGrid<CUDA>(mask),
        points(Accessor<CUDA, float, 3>::Get(points)),
        normals(Accessor<CUDA, float, 3>::Get(normals)) {}

  typename Accessor<CUDA, float, 3>::Type points;
  typename Accessor<CUDA, float, 3>::Type normals;
};

typedef PointGrid<false> CPUPointGrid;
typedef PointGrid<true> CUDAPointGrid;

template <bool CUDA>
class IntensityPointGrid : public PointGrid<CUDA> {
 public:
  IntensityPointGrid(const torch::Tensor &image,
                           const torch::Tensor &grad_image,
                           const torch::Tensor &points,
                           const torch::Tensor normals,
                           const torch::Tensor &mask)
      : PointGrid<CUDA>(points, normals, mask),
        image(Accessor<CUDA, float, 2>::Get(image)),
        grad_image(Accessor<CUDA, float, 3>::Get(grad_image)) {}

  typename Accessor<CUDA, float, 2>::Type image;
  typename Accessor<CUDA, float, 3>::Type grad_image;
};
}  // namespace

typedef IntensityPointGrid<false> CPUIntensityPointGrid;
typedef IntensityPointGrid<true> CUDAIntensityPointGrid;

}  // namespace fiontb
