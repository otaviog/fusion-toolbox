#include "surfel_fusion.hpp"

#include "accessor.hpp"
#include "eigen_common.hpp"

namespace fiontb {
template <Device dev>
struct SurfelModelAccessor {
  typename Accessor<dev, float, 2>::T positions;
  typename Accessor<dev, float, 1>::T confidences;
  typename Accessor<dev, float, 2>::T normals;
  typename Accessor<dev, float, 1>::T radii;
  typename Accessor<dev, uint8_t, 2>::T colors;
  typename Accessor<dev, int32_t, 1>::T times;
  typename Accessor<dev, float, 2>::T features;

  SurfelModelAccessor(MappedSurfelModel params)
      : positions(Accessor<dev, float, 2>::Get(params.positions)),
        confidences(Accessor<dev, float, 1>::Get(params.confidences)),
        normals(Accessor<dev, float, 2>::Get(params.normals)),
        radii(Accessor<dev, float, 1>::Get(params.radii)),
        colors(Accessor<dev, uint8_t, 2>::Get(params.colors)),
        times(Accessor<dev, int32_t, 1>::Get(params.times)),
        features(Accessor<dev, float, 2>::Get(params.features)) {}

  SurfelModelAccessor(SurfelCloud params)
      : positions(Accessor<dev, float, 2>::Get(params.positions)),
        confidences(Accessor<dev, float, 1>::Get(params.confidences)),
        normals(Accessor<dev, float, 2>::Get(params.normals)),
        radii(Accessor<dev, float, 1>::Get(params.radii)),
        colors(Accessor<dev, uint8_t, 2>::Get(params.colors)),
        times(Accessor<dev, int32_t, 1>::Get(params.times)),
        features(Accessor<dev, float, 2>::Get(params.features)) {}

  FTB_DEVICE_HOST inline Vector<float, 3> position(int idx) const {
    return to_vec3<float>(positions[idx]);
  }

  FTB_DEVICE_HOST inline void set_position(int idx, Vector<float, 3> value) {
    positions[idx][0] = value[0];
    positions[idx][1] = value[1];
    positions[idx][2] = value[2];
  }

  FTB_DEVICE_HOST inline Vector<float, 3> normal(int idx) const {
    return to_vec3<float>(normals[idx]);
  }

  FTB_DEVICE_HOST inline void set_normal(int idx, Vector<float, 3> value) {
    normals[idx][0] = value[0];
    normals[idx][1] = value[1];
    normals[idx][2] = value[2];
  }

  FTB_DEVICE_HOST inline Vector<float, 3> color(int idx) const {
    return to_vec3<float>(colors[idx]);
  }

  FTB_DEVICE_HOST inline void set_color(int idx, Vector<float, 3> value) {
    colors[idx][0] = uint8_t(value[0]);
    colors[idx][1] = uint8_t(value[1]);
    colors[idx][2] = uint8_t(value[2]);
  }
};

template <Device dev>
struct IndexMapAccessor {
  typename Accessor<dev, float, 3>::T position_confidence;
  typename Accessor<dev, float, 3>::T normal_radius;
  typename Accessor<dev, uint8_t, 3>::T color_;
  typename Accessor<dev, int32_t, 3>::T indexmap;
  typename Accessor<dev, int32_t, 2>::T linear_indexmap;

  IndexMapAccessor(const IndexMap &params)
      : position_confidence(
            Accessor<dev, float, 3>::Get(params.position_confidence)),
        normal_radius(Accessor<dev, float, 3>::Get(params.normal_radius)),
        color_(Accessor<dev, uint8_t, 3>::Get(params.color)),
        indexmap(Accessor<dev, int32_t, 3>::Get(params.indexmap)),
        linear_indexmap(
            Accessor<dev, int32_t, 2>::Get(params.indexmap.view({-1, 3}))) {}

  FTB_DEVICE_HOST inline bool empty(int row, int col) const {
    return indexmap[row][col][1] == 0;
  }

  FTB_DEVICE_HOST inline int32_t index(int row, int col) const {
    return indexmap[row][col][0];
  }

  FTB_DEVICE_HOST inline int32_t index(int idx) const {
    return linear_indexmap[idx][0];
  }

  FTB_DEVICE_HOST inline int32_t to_linear_index(int row, int col) const {
    return row * indexmap.size(1) + col;
  }

  FTB_DEVICE_HOST inline void to_rowcol_index(int linear, int *orow,
                                                 int *ocol) const {
    int row = linear / width();
    int col = linear - row * width();

    *orow = row;
    *ocol = col;
  }

  FTB_DEVICE_HOST inline int32_t time(int row, int col) const {
    return indexmap[row][col][2];
  }

  FTB_DEVICE_HOST inline Vector<float, 3> position(int row, int col) const {
    return to_vec3<float>(position_confidence[row][col]);
  }

  FTB_DEVICE_HOST inline float confidence(int row, int col) const {
    return position_confidence[row][col][3];
  }

  FTB_DEVICE_HOST inline Vector<float, 3> normal(int row, int col) const {
    return to_vec3<float>(normal_radius[row][col]);
  }

  FTB_DEVICE_HOST inline Vector<float, 3> color(int row, int col) const {
    return to_vec3<float>(color_[row][col]);
  }

  FTB_DEVICE_HOST inline float radius(int row, int col) const {
    return normal_radius[row][col][3];
  }

  FTB_DEVICE_HOST inline int width() const {
    return position_confidence.size(1);
  }

  FTB_DEVICE_HOST inline int height() const {
    return position_confidence.size(0);
  }
};
}  // namespace fiontb
