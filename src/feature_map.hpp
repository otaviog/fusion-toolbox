#include "filtering.hpp"

#include "accessor.hpp"

namespace fiontb {

template <Device dev, typename scalar_t>
struct ChannelInterpolator {
  const typename Accessor<dev, scalar_t, 3>::T feature_map;
  scalar_t u_ratio;
  scalar_t v_ratio;
  int vi, ui;

  FTB_DEVICE_HOST ChannelInterpolator(
      const typename Accessor<dev, scalar_t, 3>::T feature_map, scalar_t u,
      scalar_t v)
      : feature_map(feature_map) {
    ui = int(u);
    vi = int(v);

    u_ratio = u - scalar_t(ui);
    v_ratio = v - scalar_t(vi);
  }

  FTB_DEVICE_HOST scalar_t Get(int channel) const {
    const int width = feature_map.size(2);
    const int height = feature_map.size(1);

    const auto channel_map = feature_map[channel];

    const scalar_t val00 = channel_map[vi][ui];
    const scalar_t val10 = (ui + 1 < width) ? channel_map[vi][ui + 1] : 0;
    const scalar_t val01 = (vi + 1 < height) ? channel_map[vi + 1][ui] : 0;
    const scalar_t val11 =
        (ui + 1 < width && vi + 1 < height) ? channel_map[vi + 1][ui + 1] : 0;

    const scalar_t u0_interp = val00 * (1 - u_ratio) + val10 * u_ratio;
    const scalar_t u1_interp = val01 * (1 - u_ratio) + val11 * u_ratio;
    const scalar_t val = u0_interp * (1 - v_ratio) + u1_interp * v_ratio;
    return val;
  }
};

template <Device dev, typename scalar_t>
struct ChannelInterpolatorBackward {
  const ChannelInterpolator<dev, scalar_t> u0, u1, v0, v1;
  const scalar_t div;

  FTB_DEVICE_HOST ChannelInterpolatorBackward(
      const typename Accessor<dev, scalar_t, 3>::T feature_map, scalar_t u,
      scalar_t v, scalar_t h)
      : u0(feature_map, u + h, v),
        u1(feature_map, max(u - h, scalar_t(0)), v),
        v0(feature_map, u, max(v - h, scalar_t(0))),
        v1(feature_map, u, v + h),
        div(0.5 * h) {}

  FTB_DEVICE_HOST void Get(int channel, scalar_t &du, scalar_t &dv) const {
    du = (u0.Get(channel) - u1.Get(channel)) * div;
    dv = (v0.Get(channel) - v1.Get(channel)) * div;
  }
};

template <Device dev, typename scalar_t>
struct FeatureMap {
  const typename Accessor<dev, scalar_t, 3>::T feature_map;

  FeatureMap(const torch::Tensor &feature_map)
      : feature_map(Accessor<dev, scalar_t, 3>::Get(feature_map)) {}

  FTB_DEVICE_HOST ChannelInterpolator<dev, scalar_t> InterpolateChannel(
      scalar_t u, scalar_t v) {
    return ChannelInterpolator<dev, scalar_t>(feature_map, u, v);
  }

  FTB_DEVICE_HOST ChannelInterpolatorBackward<dev, scalar_t>
  InterpolateChannelBackward(scalar_t u, scalar_t v, scalar_t h = 0.05) {
    return ChannelInterpolatorBackward<dev, scalar_t>(feature_map, u, v, h);
  }

  FTB_DEVICE_HOST int channel_size() const { return feature_map.size(0); }
};
}  // namespace fiontb
