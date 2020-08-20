#include "nearest_neighbors.hpp"

#include "accessor.hpp"
#include "error.hpp"
#include "kernel.hpp"
#include "math.hpp"

namespace slamtb {
namespace {
const float DELTA_H = 0.0005;
}

template <Device dev, typename scalar_t>
struct ForwardKernel {
  const typename Accessor<dev, scalar_t, 2>::T nn_distances;
  const typename Accessor<dev, int64_t, 2>::T nn_index;
  const typename Accessor<dev, scalar_t, 2>::T features;
  typename Accessor<dev, scalar_t, 2>::T out_features;

  ForwardKernel(const torch::Tensor &nn_distances,
                const torch::Tensor &nn_index, const torch::Tensor &features,
                torch::Tensor out_features)
      : nn_distances(Accessor<dev, scalar_t, 2>::Get(nn_distances)),
        nn_index(Accessor<dev, int64_t, 2>::Get(nn_index)),
        features(Accessor<dev, scalar_t, 2>::Get(features)),
        out_features(Accessor<dev, scalar_t, 2>::Get(out_features)) {}

  STB_DEVICE_HOST void operator()(int idx) {
    scalar_t total_similarity = 0;

    int num_neighbors = 0;
    for (int k = 0; k < nn_distances.size(1); ++k) {
      const scalar_t distance = nn_distances[idx][k];
      if (distance < NumericLimits<dev, scalar_t>::infinity()) {
        total_similarity += 1 / (1 + distance);
        ++num_neighbors;
      }
    }

    for (int channel = 0; channel < features.size(0); ++channel) {
      scalar_t feature = 0;
      for (int k = 0; k < nn_index.size(1); ++k) {
        const int64_t index = nn_index[idx][k];
        if (index >= features.size(1)) break;

        const scalar_t distance = nn_distances[idx][k];
        const scalar_t similarity = 1 / (1 + distance);

        scalar_t weight = similarity / total_similarity;
        feature += features[channel][index] * weight;
      }
      out_features[channel][idx] = feature;
    }
  }
};

void NearestNeighborsOp::Forward(const torch::Tensor &nn_distances,
                                 const torch::Tensor &nn_index,
                                 const torch::Tensor &features,
                                 torch::Tensor out_features) {
  const auto ref_device = nn_distances.device();
  const auto ref_type = nn_distances.scalar_type();

  STB_CHECK_DEVICE(ref_device, nn_index);
  STB_CHECK_DEVICE(ref_device, features);
  STB_CHECK_DEVICE(ref_device, out_features);

  if (ref_device.is_cuda()) {
    AT_DISPATCH_FLOATING_TYPES(
        ref_type, "NearestNeighborsOp::Forward", ([&] {
          ForwardKernel<kCUDA, scalar_t> kernel(nn_distances, nn_index,
                                                features, out_features);
          Launch1DKernelCUDA(kernel, out_features.size(1));
        }));
  } else {
    AT_DISPATCH_FLOATING_TYPES(
        ref_type, "NearestNeighborsOp::Forward", ([&] {
          ForwardKernel<kCPU, scalar_t> kernel(nn_distances, nn_index, features,
                                               out_features);
          Launch1DKernelCPU(kernel, out_features.size(1));
        }));
  }
}

template <Device dev, typename scalar_t>
struct EpsilonDistancesKernel {
  const typename Accessor<dev, scalar_t, 2>::T target_xyz;
  const typename Accessor<dev, scalar_t, 2>::T source_xyz;
  const typename Accessor<dev, int64_t, 2>::T nn_index;

  typename Accessor<dev, scalar_t, 3>::T epsilon_distances;

  const int64_t max_index;
  scalar_t xyz_epsilon[6][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0},
                                {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};

  EpsilonDistancesKernel(const torch::Tensor &target_xyz,
                         const torch::Tensor &source_xyz,
                         const torch::Tensor &nn_index,
                         const torch::Tensor &epsilon_distances,
                         const scalar_t h)
      : target_xyz(Accessor<dev, scalar_t, 2>::Get(target_xyz)),
        source_xyz(Accessor<dev, scalar_t, 2>::Get(source_xyz)),
        nn_index(Accessor<dev, int64_t, 2>::Get(nn_index)),
        epsilon_distances(Accessor<dev, scalar_t, 3>::Get(epsilon_distances)),
        max_index(target_xyz.size(0) - 1) {
    xyz_epsilon[0][0] = h;
    xyz_epsilon[1][0] = -h;
    xyz_epsilon[2][1] = h;
    xyz_epsilon[3][1] = -h;
    xyz_epsilon[4][2] = h;
    xyz_epsilon[5][2] = -h;
  }

  STB_DEVICE_HOST void operator()(int row, int col) {
    const Vector<scalar_t, 3> xyz =
        Vector<scalar_t, 3>(xyz_epsilon[col][0], xyz_epsilon[col][1],
                            xyz_epsilon[col][2]) +
        to_vec3<scalar_t>(source_xyz[row]);

    scalar_t total_similarity = 0;
    int num_neighbors = 0;
    for (int k = 0; k < nn_index.size(1); ++k) {
      const int64_t index = nn_index[row][k];
      if (index > max_index) break;

      const Vector<scalar_t, 3> tgt_xyz = to_vec3<scalar_t>(target_xyz[index]);
      const scalar_t distance = (tgt_xyz - xyz).squaredNorm();

      const scalar_t similarity = 1 / (1 + distance);
      epsilon_distances[row][col][k] = similarity;
      total_similarity += similarity;

      ++num_neighbors;
    }

    if (num_neighbors == 0) {
      epsilon_distances[row][col][0] = 1;
      return;
    }

    for (int k = 0; k < nn_index.size(1); ++k) {
      const int64_t index = nn_index[row][k];
      if (index > max_index) break;
      const scalar_t similarity = epsilon_distances[row][col][k];
      epsilon_distances[row][col][k] = similarity / total_similarity;
    }
  }
};

void NearestNeighborsOp::ComputeEpsilonDistances(
    const torch::Tensor &target_xyz, const torch::Tensor &source_xyz,
    const torch::Tensor &nn_index, const torch::Tensor &epsilon_distances) {
  const auto ref_device = target_xyz.device();
  const auto ref_type = target_xyz.scalar_type();
  STB_CHECK_DEVICE(ref_device, source_xyz);
  STB_CHECK_DEVICE(ref_device, nn_index);
  STB_CHECK_DEVICE(ref_device, epsilon_distances);

  if (ref_device.is_cuda()) {
    AT_DISPATCH_FLOATING_TYPES(ref_type, "EpsilonDistancesKernel", ([&] {
                                 EpsilonDistancesKernel<kCUDA, scalar_t> kernel(
                                     target_xyz, source_xyz, nn_index,
                                     epsilon_distances, scalar_t(DELTA_H));
                                 Launch2DKernelCUDA(kernel,
                                                    epsilon_distances.size(1),
                                                    epsilon_distances.size(0));
                               }));
  } else {
    AT_DISPATCH_FLOATING_TYPES(ref_type, "EpsilonDistancesKernel", ([&] {
                                 EpsilonDistancesKernel<kCPU, scalar_t> kernel(
                                     target_xyz, source_xyz, nn_index,
                                     epsilon_distances, scalar_t(DELTA_H));
                                 Launch2DKernelCPU(kernel,
                                                   epsilon_distances.size(1),
                                                   epsilon_distances.size(0));
                               }));
  }
}

template <Device dev, typename scalar_t>
struct NNGrad {
  const typename Accessor<dev, scalar_t, 2>::Ts epsilon_distances;
  const typename Accessor<dev, int64_t, 1>::Ts nn_index;
  const typename Accessor<dev, scalar_t, 2>::T features;
  const scalar_t div;
  const int64_t max_index;

  STB_DEVICE_HOST NNGrad(
      const typename Accessor<dev, scalar_t, 2>::Ts epsilon_distances,
      const typename Accessor<dev, int64_t, 1>::Ts nn_index,
      const typename Accessor<dev, scalar_t, 2>::T features, scalar_t delta_h)
      : epsilon_distances(epsilon_distances),
        nn_index(nn_index),
        features(features),
        div(scalar_t(1) / (2 * delta_h)),
        max_index(features.size(1)) {}

  STB_DEVICE_HOST inline void GetGrad(int channel, scalar_t &dx, scalar_t &dy,
                                      scalar_t &dz) const {
    dx = (Get(channel, 0) - Get(channel, 1)) * div;
    dy = (Get(channel, 2) - Get(channel, 3)) * div;
    dz = (Get(channel, 4) - Get(channel, 5)) * div;
  }

 private:
  STB_DEVICE_HOST inline scalar_t Get(int channel, int which_epsilon) const {
    const auto distance_weights = epsilon_distances[which_epsilon];
    scalar_t feature_avg = 0;

    for (int k = 0; k < nn_index.size(0); ++k) {
      const int64_t index = nn_index[k];
      if (index < max_index) {
        feature_avg += features[channel][index] * distance_weights[k];
      }
    }

    return feature_avg;
  }
};

template <Device dev, typename scalar_t>
struct BackwardKernel {
  const typename Accessor<dev, scalar_t, 3>::T epsilon_distances;
  const typename Accessor<dev, int64_t, 2>::T nn_index;
  const typename Accessor<dev, scalar_t, 2>::T features;
  const typename Accessor<dev, scalar_t, 2>::T dl_features;

  const scalar_t delta_h;

  typename Accessor<dev, scalar_t, 2>::T dl_xyz;

  BackwardKernel(const torch::Tensor &epsilon_distances,
                 const torch::Tensor &nn_index, const torch::Tensor &features,
                 const torch::Tensor &dl_features, torch::Tensor dl_xyz,
                 const scalar_t delta_h)
      : epsilon_distances(Accessor<dev, scalar_t, 3>::Get(epsilon_distances)),
        nn_index(Accessor<dev, int64_t, 2>::Get(nn_index)),
        features(Accessor<dev, scalar_t, 2>::Get(features)),
        dl_features(Accessor<dev, scalar_t, 2>::Get(dl_features)),
        delta_h(delta_h),
        dl_xyz(Accessor<dev, scalar_t, 2>::Get(dl_xyz)) {}

  STB_DEVICE_HOST void operator()(int idx) {
    const NNGrad<dev, scalar_t> backward(epsilon_distances[idx], nn_index[idx],
                                         features, delta_h);
    scalar_t dl_x = 0, dl_y = 0, dl_z = 0;
    for (int channel = 0; channel < features.size(0); ++channel) {
      scalar_t dx, dy, dz;
      backward.GetGrad(channel, dx, dy, dz);

      const scalar_t dl_feature = dl_features[channel][idx];
      dl_x += dx * dl_feature;
      dl_y += dy * dl_feature;
      dl_z += dz * dl_feature;
    }

    dl_xyz[idx][0] = dl_x;
    dl_xyz[idx][1] = dl_y;
    dl_xyz[idx][2] = dl_z;
  }
};

void NearestNeighborsOp::Backward(const torch::Tensor &epsilon_distances,
                                  const torch::Tensor &nn_index,
                                  const torch::Tensor &features,
                                  const torch::Tensor &dl_features,
                                  torch::Tensor dl_xyz) {
  const auto ref_device = epsilon_distances.device();
  const auto ref_type = epsilon_distances.scalar_type();

  STB_CHECK_DEVICE(ref_device, epsilon_distances);
  STB_CHECK_DEVICE(ref_device, nn_index);
  STB_CHECK_DEVICE(ref_device, features);
  STB_CHECK_DEVICE(ref_device, dl_features);
  STB_CHECK_DEVICE(ref_device, dl_xyz);

  if (ref_device.is_cuda()) {
    AT_DISPATCH_FLOATING_TYPES(ref_type, "NearestNeighborsOp::Backward", ([&] {
                                 BackwardKernel<kCUDA, scalar_t> kernel(
                                     epsilon_distances, nn_index, features,
                                     dl_features, dl_xyz, DELTA_H);
                                 Launch1DKernelCUDA(kernel, dl_xyz.size(0));
                               }));
  } else {
    AT_DISPATCH_FLOATING_TYPES(ref_type, "NearestNeighborsOp::Backward", ([&] {
                                 BackwardKernel<kCPU, scalar_t> kernel(
                                     epsilon_distances, nn_index, features,
                                     dl_features, dl_xyz, DELTA_H);
                                 Launch1DKernelCPU(kernel, dl_xyz.size(0));
                               }));
  }
}

}  // namespace slamtb
