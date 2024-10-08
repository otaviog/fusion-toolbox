#include "surfel_fusion.hpp"

#include <mutex>

#include "accessor.hpp"
#include "eigen_common.hpp"

namespace slamtb {
template <Device dev>
struct SurfelModelAccessor {
  typename Accessor<dev, float, 2>::T points;
  typename Accessor<dev, float, 1>::T confidences;
  typename Accessor<dev, float, 2>::T normals;
  typename Accessor<dev, float, 1>::T radii;
  typename Accessor<dev, uint8_t, 2>::T colors;
  typename Accessor<dev, int32_t, 1>::T times;
  typename Accessor<dev, float, 2>::T features;

  SurfelModelAccessor(MappedSurfelModel params)
      : points(Accessor<dev, float, 2>::Get(params.points)),
        confidences(Accessor<dev, float, 1>::Get(params.confidences)),
        normals(Accessor<dev, float, 2>::Get(params.normals)),
        radii(Accessor<dev, float, 1>::Get(params.radii)),
        colors(Accessor<dev, uint8_t, 2>::Get(params.colors)),
        times(Accessor<dev, int32_t, 1>::Get(params.times)),
        features(Accessor<dev, float, 2>::Get(params.features)) {}

  SurfelModelAccessor(SurfelCloud params)
      : points(Accessor<dev, float, 2>::Get(params.points)),
        confidences(Accessor<dev, float, 1>::Get(params.confidences)),
        normals(Accessor<dev, float, 2>::Get(params.normals)),
        radii(Accessor<dev, float, 1>::Get(params.radii)),
        colors(Accessor<dev, uint8_t, 2>::Get(params.colors)),
        times(Accessor<dev, int32_t, 1>::Get(params.times)),
        features(Accessor<dev, float, 2>::Get(params.features)) {}

  STB_DEVICE_HOST inline Vector<float, 3> point(int idx) const {
    return to_vec3<float>(points[idx]);
  }

  STB_DEVICE_HOST inline void set_point(int idx, Vector<float, 3> value) {
    points[idx][0] = value[0];
    points[idx][1] = value[1];
    points[idx][2] = value[2];
  }

  STB_DEVICE_HOST inline Vector<float, 3> normal(int idx) const {
    return to_vec3<float>(normals[idx]);
  }

  STB_DEVICE_HOST inline void set_normal(int idx, Vector<float, 3> value) {
    normals[idx][0] = value[0];
    normals[idx][1] = value[1];
    normals[idx][2] = value[2];
  }

  STB_DEVICE_HOST inline Vector<float, 3> color(int idx) const {
    return to_vec3<float>(colors[idx]);
  }

  STB_DEVICE_HOST inline void set_color(int idx, Vector<float, 3> value) {
    colors[idx][0] = uint8_t(value[0]);
    colors[idx][1] = uint8_t(value[1]);
    colors[idx][2] = uint8_t(value[2]);
  }
};

template <Device dev>
using SurfelCloudAccessor = SurfelModelAccessor<dev>;

template <Device dev>
struct IndexMapAccessor {
  typename Accessor<dev, float, 3>::T point_confidence;
  typename Accessor<dev, float, 3>::T normal_radius;
  typename Accessor<dev, int32_t, 3>::T indexmap;
  typename Accessor<dev, int32_t, 2>::T linear_indexmap;

  IndexMapAccessor(const IndexMap &params)
      : point_confidence(Accessor<dev, float, 3>::Get(params.point_confidence)),
        normal_radius(Accessor<dev, float, 3>::Get(params.normal_radius)),
        indexmap(Accessor<dev, int32_t, 3>::Get(params.indexmap)),
        linear_indexmap(
            Accessor<dev, int32_t, 2>::Get(params.indexmap.view({-1, 3}))) {}

  STB_DEVICE_HOST inline bool empty(int row, int col) const {
    return indexmap[row][col][1] == 0;
  }

  STB_DEVICE_HOST inline int32_t index(int row, int col) const {
    return indexmap[row][col][0];
  }

  STB_DEVICE_HOST inline int32_t index(int idx) const {
    return linear_indexmap[idx][0];
  }

  STB_DEVICE_HOST inline int32_t to_linear_index(int row, int col) const {
    return row * indexmap.size(1) + col;
  }

  STB_DEVICE_HOST inline void to_rowcol_index(int linear, int *orow,
                                              int *ocol) const {
    int row = linear / width();
    int col = linear - row * width();

    *orow = row;
    *ocol = col;
  }

  STB_DEVICE_HOST inline int32_t time(int row, int col) const {
    return indexmap[row][col][2];
  }

  STB_DEVICE_HOST inline Vector<float, 3> point(int row, int col) const {
    return to_vec3<float>(point_confidence[row][col]);
  }

  STB_DEVICE_HOST inline float confidence(int row, int col) const {
    return point_confidence[row][col][3];
  }

  STB_DEVICE_HOST inline Vector<float, 3> normal(int row, int col) const {
    return to_vec3<float>(normal_radius[row][col]);
  }

  STB_DEVICE_HOST inline float radius(int row, int col) const {
    return normal_radius[row][col][3];
  }

  STB_DEVICE_HOST inline int width() const { return point_confidence.size(1); }

  STB_DEVICE_HOST inline int height() const { return point_confidence.size(0); }
};

}  // namespace slamtb
