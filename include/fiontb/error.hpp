#pragma once

#include <sstream>
#include <torch/torch.h>

namespace fiontb {

inline void Check(bool test, const char *file, int line, const char *message) {
  if (!test) {
    std::stringstream msg;
    msg << "Check failed: " << file << "(" << line << "): " << message;
    throw std::runtime_error(msg.str());
  }
}

inline void CheckDevice(const torch::Device expected_dev,
                        const torch::Tensor &test_tensor, const char *file,
                        int line) {
  if (expected_dev != test_tensor.device()) {
    if (test_tensor.is_cuda()) {
      Check(false, file, line, "Expected a cpu tensor");
    } else {
      Check(false, file, line, "Expected a gpu tensor");
    }
  }
}



}  // namespace fiontb

#define FTB_CHECK(test, msg) fiontb::Check(test, __FILE__, __LINE__, msg)
#define FTB_CHECK_DEVICE(device, test_tensor) \
  fiontb::CheckDevice(device, test_tensor, __FILE__, __LINE__)
