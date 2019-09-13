#include "camera.hpp"

namespace fiontb {

template<Device dev>
struct ProjectKernel {
  KCamera<dev> kcam;
  const typename Accessor<dev, float, 3>::T points;
  typename Accessor<dev, float, 3>::T out_projection;
  
  void operator()(int i) {
    const Eigen::Vector3f point = to_vec3<float>(points[i]);
    float x, y;
    kcam.Project(point, x, y);
    
    out_projection[i][0] = x;
    out_projection[i][1] = y;
  }  
};


torch::Tensor KCameraGradOp::Project(torch::Tensor points,
                                     torch::Tensor intrinsic_matrix) {
  KCamera<kCPU> kcam(intrinsic_matrix);

  auto array_points = points.view(-1, 3);
  //for (points.
  kcam.proje
}

struct ProjectBackwardKernel {
  
};

torch::Tensor KCameraGradOp::ProjectBackward(const torch::Tensor points,
                                             torch::Tensor intrinsic_matrix) {}
}  // namespace fiontb
