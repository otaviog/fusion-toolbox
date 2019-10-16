#include "camera.hpp"

#include <pybind11/eigen.h>
#include <torch/csrc/utils/pybind.h>

#include "error.hpp"

namespace fiontb {
void ProjectOp::RegisterPybind(pybind11::module &m) {
  pybind11::class_<ProjectOp>(m, "ProjectOp")
      .def_static("forward", &ProjectOp::Forward)
      .def_static("backward", &ProjectOp::Backward);
}

void RigidTransformOp::Rodrigues(const torch::Tensor &rot_matrix,
                                 torch::Tensor rodrigues) {
  FTB_CHECK(!rot_matrix.is_cuda(), "Rodrigues is cpu only");
  FTB_CHECK(!rodrigues.is_cuda(), "Rodrigues is cpu only");

  AT_DISPATCH_ALL_TYPES(rot_matrix.scalar_type(), "Rodrigues", [&] {
    const torch::TensorAccessor<scalar_t, 2> rot_acc =
        rot_matrix.accessor<scalar_t, 2>();
    torch::TensorAccessor<scalar_t, 1> rodrigues_acc =
        rodrigues.accessor<scalar_t, 1>();

    Eigen::Matrix3d rot;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        rot(0, 0) = rot_acc[i][j];
      }
    }

    const auto axis = Eigen::AngleAxisd(rot).axis();
    rodrigues[0] = axis[0];
    rodrigues[1] = axis[1];
    rodrigues[2] = axis[2];
  });
}

void RigidTransformOp::RegisterPybind(pybind11::module &m) {
  pybind11::class_<RigidTransformOp>(m, "RigidTransformOp")
      .def_static("rodrigues", &RigidTransformOp::Rodrigues);
}

}  // namespace fiontb
