#include "surfel.hpp"

#include <torch/csrc/utils/pybind.h>

namespace py = pybind11;

namespace fiontb {

SurfelAllocator::SurfelAllocator(int max_surfels) {
  max_surfels_ = max_surfels;
  FreeAll();
};

void SurfelAllocator::RegisterPybind(pybind11::module &m) {
  py::class_<SurfelAllocator>(m, "SurfelAllocator")
      .def(py::init<int>())
      .def("allocate", &SurfelAllocator::Allocate)
      .def("free", &SurfelAllocator::Free)
      .def("free_all", &SurfelAllocator::FreeAll)
      .def("copy_", &SurfelAllocator::Copy_)
      .def_property("max_size", &SurfelAllocator::get_max_size, nullptr)
      .def_property("allocated_size", &SurfelAllocator::get_allocated_size,
                    nullptr)
      .def_property("free_size", &SurfelAllocator::get_free_size, nullptr);
}

template <typename scalar_t>
struct AccessorIterator: public std::iterator<std::input_iterator_tag, scalar_t>{
  torch::TensorAccessor<scalar_t, 1> accessor;
  int pos;

  AccessorIterator(torch::Tensor tensor, int pos = 0)
      : accessor(tensor.accessor<scalar_t, 1>()), pos(pos) {}

  AccessorIterator(torch::TensorAccessor<scalar_t, 1> accessor, int pos = 0)
      : accessor(accessor), pos(pos) {}

  AccessorIterator<scalar_t> &operator++() {
    pos++;
    return *this;
  }

  AccessorIterator<scalar_t> &operator--() { pos--; }

  scalar_t &operator*() { return accessor[pos]; }

  bool operator==(const AccessorIterator<scalar_t> &rhs) const {
    return pos == rhs.pos;
  }

  bool operator!=(const AccessorIterator<scalar_t> &rhs) const {
    return pos != rhs.pos;
  }

  void End() { return AccessorIterator<scalar_t>(accessor, accessor.size(0)); }
};

void SurfelAllocator::Allocate(torch::Tensor out_free_indices) {
  torch::TensorAccessor<int64_t, 1> out_free_acc(
      out_free_indices.accessor<int64_t, 1>());

  const int size_unact = out_free_indices.size(0);

  auto erase_begin = free_indices_.rbegin();
  auto erase_end = free_indices_.rbegin() + size_unact;

  std::copy(erase_begin, erase_end,
            AccessorIterator<int64_t>(out_free_indices));
  //free_indices_.erase(erase_begin, erase_end);
  free_indices_.resize(free_indices_.size() - size_unact);
#if 0
  for (int i = 0; i < size_unact; ++i) {
    const int free_idx = free_indices_.back();
    free_indices_.pop_back();
    out_free_indices[i] = free_idx;
  }
#endif
}

void SurfelAllocator::Free(const torch::Tensor &indices) {
  const torch::TensorAccessor<int64_t, 1> indices_acc(
      indices.accessor<int64_t, 1>());

  for (int i = 0; i < indices.size(0); ++i) {
    int64_t free_indice = indices_acc[i];
    free_indices_.push_back(free_indice);
  }
}

void SurfelAllocator::FreeAll() {
  free_indices_.clear();
  // free_indices_.reserve(max_surfels_);
  for (int i = 0; i < max_surfels_; ++i) {
    free_indices_.push_back(i);
  }
}

void SurfelAllocator::Copy_(const SurfelAllocator &other) {
  free_indices_ = other.free_indices_;
  max_surfels_ = other.max_surfels_;
}
}  // namespace fiontb
