#pragma once

#include <map>

#include <torch/csrc/utils/pybind.h>
#include <torch/torch.h>

namespace slamtb {
/**
 * Represents spatial features associated to integer id, such an index
 * or point id.
 */
class SparseFeatureSet {
 public:
  /**
   * Feature item.
   */
  struct Item {
    Item() {}
    Item(float weight, torch::Tensor feature)
        : weight(weight), feature(feature) {}

    /**
     * Sums the weights and the features.
     *
     * @params src the other feature item.
     */
    void Merge(const Item &src) {
      const auto src_acc = src.feature.accessor<float, 1>();
      auto dst_acc = feature.accessor<float, 1>();

      weight += src.weight;

      for (int c = 0; c < std::min(feature.size(0), src.feature.size(0)); ++c) {
        dst_acc[c] += src_acc[c];
      }
    }

    /**
     * Creates a copy of the current instance.
     */
    Item Clone() const { return Item(weight, feature.clone()); }

    float weight; /**< Factor to divide the by feature*/
    torch::Tensor feature; /**< Feature. Float [C]*/
  };

  typedef std::map<int, Item> MapType; /**<Dictionaty type to
                                        * associate the indices to featurees*/

  /**
   * Register into python.
   */
  static void RegisterPybind(pybind11::module &m);

  SparseFeatureSet() {}
  
  /**
   * Creates from keypoints originating from a grid, with its
   * dimensions determined by its mask. The point index will account
   * skipping non-masked elements.
   *
   * @param keypoint_xy X and Y key point coordinates. Tensor [Nx2]
   * int32.
   * @param features Features. Tensor [NxC] float32.
   * @param mask Mask image. Tensor [HxW] bool.
   */
  SparseFeatureSet(const torch::Tensor &keypoint_xy, const torch::Tensor &features,
                   const torch::Tensor &mask);

  /**
   * Merge intern features. 
   *
   * @param correspondences Pairs of target and source merges. Tensor
   * [Nx2] int64.
   */
  void Merge(const torch::Tensor &correspondences);

  /**
   * Merge with external features. 
   *
   * @param correspondences Pairs of target and source merges. Tensor
   * [Nx2] int64.
   * @param source Source feature set.
   */
  void Merge(const torch::Tensor &correspondences,
             const SparseFeatureSet &source);

  /**
   * Add features to this set.
   *
   * @param dense_index Mapping between source index to new ones.
   * Tensor [N] int64.
   * @param source_set Source features.
   */
  void Add(const torch::Tensor &dense_index, const SparseFeatureSet &source_set);

  /**
   * Remove features by its indices.
   * @param index indices to remove.
   */
  void Remove(const torch::Tensor &index);

  /**
   * Clear all features.
   */
  void Clear() {
    feature_map_.clear();
  }

  /**
   * Returns a tensor of features' indices.
   * @return Indices. Tensor [N] int64.
   */
  torch::Tensor GetIndex() const;

  /**
   * Returns a tensor with all features concatenated. Same order as GetIndex.
   * @return Features. Tensor [NxC] float.
   */
  torch::Tensor GetFeatures() const;

  /**
   * Returns a tensor with all weights concatenated. Same order as
   * GetIndex.
   * @return Weights. Tensor [N] float.
   */
  torch::Tensor GetWeights() const;

  /**
   * Slices the sparse feature according to one dimension selector.
   *
   * @param Selector. If it's a [N] bool tensor, then it will iterate
   * over the tensor, selecting the masked one. If it's [M] int64,
   * then it will return the ones in the selector.
   *
   * @return Sliced set.
   */
  std::shared_ptr<SparseFeatureSet> Select(const torch::Tensor &selector);

  /**
   * Convert to a Python' dictionary.
   * @return Dictionary.
   */
  pybind11::dict AsDict() const;

  /**
   * Loads from a dict. 
   * @param dict Python's dictionary.
   */
  void FromDict(const pybind11::dict &dict);

  std::string __str__() const;

  int __len__() const;

 private:
  MapType feature_map_;
};
}  // namespace slamtb
