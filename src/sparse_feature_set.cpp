#include "sparse_feature_set.hpp"
#include <sstream>

namespace py = pybind11;

namespace slamtb {

void SparseFeatureSet::RegisterPybind(py::module &m) {
  py::class_<SparseFeatureSet, std::shared_ptr<SparseFeatureSet>>(
      m, "SparseFeatureSet")
      .def(py::init<>())
      .def(py::init<const torch::Tensor &, const torch::Tensor &,
                    const torch::Tensor &>())
      .def("add", &SparseFeatureSet::Add)
      .def("remove", &SparseFeatureSet::Remove)
      .def("clear", &SparseFeatureSet::Clear)
      .def("merge", (void (SparseFeatureSet::*)(const torch::Tensor &,
                                                const SparseFeatureSet &)) &
                        SparseFeatureSet::Merge)
      .def("merge", (void (SparseFeatureSet::*)(const torch::Tensor &)) &
                        SparseFeatureSet::Merge)
      .def("select", &SparseFeatureSet::Select)
      .def("get_index", &SparseFeatureSet::GetIndex)
      .def("get_features", &SparseFeatureSet::GetFeatures)
      .def("get_weights", &SparseFeatureSet::GetWeights)

      .def("as_dict", &SparseFeatureSet::AsDict)
      .def("from_dict", &SparseFeatureSet::FromDict)
      .def(py::pickle(
          [](const SparseFeatureSet &self) {  //__getstate__
            return self.AsDict();
          },
          [](py::dict dict) {  // __setstate__
            std::shared_ptr<SparseFeatureSet> self(new SparseFeatureSet);
            self->FromDict(dict);
            return self;
          }))
      .def("__str__", &SparseFeatureSet::__str__)
      .def("__repr__", &SparseFeatureSet::__str__)
      .def("__len__", &SparseFeatureSet::__len__);
}

SparseFeatureSet::SparseFeatureSet(const torch::Tensor &keypoint_xy,
                                   const torch::Tensor &features,
                                   const torch::Tensor &mask) {
  const torch::TensorAccessor<int32_t, 2> xy_acc =
      keypoint_xy.accessor<int32_t, 2>();
  const torch::TensorAccessor<float, 2> feat_acc =
      features.accessor<float, 2>();
  const torch::TensorAccessor<bool, 2> mask_acc = mask.accessor<bool, 2>();

  // The point id must match with the point in the point-cloud.
  // So it will skip the mask empty values.

  torch::Tensor point_id_grid =
      torch::full(mask.sizes(), -1, torch::TensorOptions(torch::kInt32));
  torch::TensorAccessor<int32_t, 2> point_id_grid_acc =
      point_id_grid.accessor<int32_t, 2>();
  int point_id = 0;
  for (int row = 0; row < point_id_grid.size(0); ++row) {
    for (int col = 0; col < point_id_grid.size(1); ++col) {
      if (!mask_acc[row][col]) continue;
      point_id_grid_acc[row][col] = point_id++;
    }
  }

  for (size_t i = 0; i < xy_acc.size(0); ++i) {
    const int32_t x = xy_acc[i][0];
    const int32_t y = xy_acc[i][1];

    assert(y < point_id_grid.size(0));
    assert(x < point_id_grid.size(1));

    if (!mask_acc[y][x]) continue;

    feature_map_[point_id_grid_acc[y][x]] =
        Item(1.0f, features.slice(0, i, i + 1).clone().squeeze());
  }
}

void SparseFeatureSet::Merge(const torch::Tensor &correspondences,
                             const SparseFeatureSet &source) {
  torch::TensorAccessor<int64_t, 2> corresp_acc =
      correspondences.accessor<int64_t, 2>();

  for (int corresp = 0; corresp < correspondences.size(0); ++corresp) {
    const int64_t dst_index = corresp_acc[corresp][0];
    const int64_t src_index = corresp_acc[corresp][1];

    const auto src_it = source.feature_map_.find(src_index);
    if (src_it == source.feature_map_.end()) {
      continue;
    }

    auto dst_it = feature_map_.find(dst_index);
    if (dst_it == feature_map_.end()) {
      feature_map_[dst_index] = src_it->second.Clone();
      continue;
    }

    dst_it->second.Merge(src_it->second);
  }
}

void SparseFeatureSet::Merge(const torch::Tensor &correspondences) {
  const torch::TensorAccessor<int64_t, 2> acc =
      correspondences.accessor<int64_t, 2>();
  for (int i = 0; i < correspondences.size(0); ++i) {
    const int64_t dst_id = acc[i][0];
    const int64_t src_id = acc[i][1];

    const auto src_it = feature_map_.find(src_id);
    if (src_it == feature_map_.end()) {
      continue;
    }

    auto dst_it = feature_map_.find(dst_id);
    if (dst_it == feature_map_.end()) {
      feature_map_[dst_id] = src_it->second.Clone();
      continue;
    }

    dst_it->second.Merge(src_it->second);
    feature_map_.erase(src_it);
  }
}

void SparseFeatureSet::Add(const torch::Tensor &dense_index,
                           const SparseFeatureSet &source) {
  const torch::TensorAccessor<int64_t, 1> index_acc =
      dense_index.accessor<int64_t, 1>();

  for (auto item : source.feature_map_) {
    feature_map_[index_acc[item.first]] = item.second.Clone();
  }
}

void SparseFeatureSet::Remove(const torch::Tensor &index) {
  const torch::TensorAccessor<int64_t, 1> index_acc =
      index.accessor<int64_t, 1>();

  for (size_t i = 0; i < index.size(0); ++i) {
    feature_map_.erase(index_acc[i]);
  }
}

std::shared_ptr<SparseFeatureSet> SparseFeatureSet::Select(
    const torch::Tensor &selector) {
  std::shared_ptr<SparseFeatureSet> new_set(new SparseFeatureSet);

  if (selector.dtype() == torch::kBool) {
    int point_id = 0;
    const torch::TensorAccessor<bool, 1> bool_sel_acc =
        selector.accessor<bool, 1>();

    for (int i = 0; i < selector.size(0); ++i) {
      if (!bool_sel_acc[i]) continue;
      const int this_point_id = point_id++;
      const auto find_iter = feature_map_.find(i);
      if (find_iter == feature_map_.end()) continue;

      new_set->feature_map_[this_point_id] = find_iter->second.Clone();
    }
  } else if (selector.dtype() == torch::kInt64) {
    const torch::TensorAccessor<int64_t, 1> index_sel_acc =
        selector.accessor<int64_t, 1>();
    for (int i = 0; i < selector.size(0); ++i) {
      const int64_t index = index_sel_acc[i];
      const auto find_iter = feature_map_.find(index);
      if (find_iter == feature_map_.end()) continue;

      new_set->feature_map_[i] = find_iter->second.Clone();
    }
  } else {
    throw std::runtime_error("Invalid selector to SparseFeatureSet::Select");
  }

  return new_set;
}

torch::Tensor SparseFeatureSet::GetIndex() const {
  torch::Tensor keys =
      torch::empty({int64_t(feature_map_.size())}, torch::kInt64);
  auto keys_acc = keys.accessor<int64_t, 1>();
  int count = 0;
  for (auto item : feature_map_) {
    keys_acc[count++] = item.first;
  }

  return keys;
}

torch::Tensor SparseFeatureSet::GetFeatures() const {
  if (feature_map_.begin() == feature_map_.end()) {
    throw std::runtime_error("Empty sparse feature");
  }

  const int64_t feature_size = feature_map_.begin()->second.feature.size(0);

  torch::Tensor features = torch::empty(
      {int64_t(feature_map_.size()), feature_size}, torch::kFloat32);
  auto features_acc = features.accessor<float, 2>();

  int row = 0;
  for (auto item : feature_map_) {
    const torch::TensorAccessor<float, 1> feat_acc =
        item.second.feature.accessor<float, 1>();
    for (int channel = 0; channel < feature_size; ++channel) {
      features_acc[row][channel] = feat_acc[channel] / item.second.weight;
    }
    ++row;
  }

  return features;
}

torch::Tensor SparseFeatureSet::GetWeights() const {
  torch::Tensor weights =
      torch::empty({int64_t(feature_map_.size())}, torch::kFloat32);
  auto weights_acc = weights.accessor<float, 1>();

  int row = 0;
  for (auto item : feature_map_) {
    weights_acc[row++] = item.second.weight;
  }

  return weights;
}

py::dict SparseFeatureSet::AsDict() const {
  py::dict dict;
  for (auto item : feature_map_) {
    dict[py::cast(item.first)] =
        py::make_tuple(item.second.weight, item.second.feature);
  }

  return dict;
}

void SparseFeatureSet::FromDict(const py::dict &dict) {
  feature_map_.clear();
  for (auto item : dict) {
    const int key = py::cast<int>(item.first);
    py::tuple value = py::cast<py::tuple>(item.second);

    const float weight = py::cast<float>(value[0]);
    torch::Tensor feature = py::cast<torch::Tensor>(value[1]);

    feature_map_[key] = Item(weight, feature);
  }
}

std::string SparseFeatureSet::__str__() const {
  std::stringstream stream;

  stream << "Sparse feature set with " << feature_map_.size() << " points";

  return stream.str();
}

int SparseFeatureSet::__len__() const { return int(feature_map_.size()); }

}  // namespace slamtb
