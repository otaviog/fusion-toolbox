#include "surfel.hpp"

#include <torch/csrc/utils/pybind.h>

namespace py = pybind11;

namespace fiontb {

SurfelAllocator::SurfelAllocator(int size, const std::string &device) {
  free_mask_byte =
      torch::ones({size}, torch::TensorOptions(torch::kUInt8).device(device));
  free_mask_bit = torch::ones({size}, torch::TensorOptions(torch::kBool));
};

void SurfelAllocator::RegisterPybind(pybind11::module &m) {
  py::class_<SurfelAllocator>(m, "SurfelAllocator")
      .def(py::init<int, std::string>())
      .def("find_unactive", &SurfelAllocator::FindUnactive)
      .def_readwrite("free_mask_byte", &SurfelAllocator::free_mask_byte)
      .def_readwrite("free_mask_bit", &SurfelAllocator::free_mask_bit);
}

void SurfelAllocator::FindUnactive(torch::Tensor unactive_indices) {
  const torch::TensorAccessor<bool, 1> free_mask_acc(
      free_mask_bit.accessor<bool, 1>());
  torch::TensorAccessor<int64_t, 1> unactive_out_acc(
      unactive_indices.accessor<int64_t, 1>());

  const int size_unact = unactive_indices.size(0);
  int found_count = 0;

  for (int i = 0; i < free_mask_bit.size(0); ++i) {
    if (free_mask_acc[i]) {
      unactive_out_acc[found_count++] = i;
      if (found_count == size_unact) break;
    }
  }
}

}  // namespace fiontb
