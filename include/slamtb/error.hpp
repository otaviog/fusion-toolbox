#pragma once

#include <sstream>
#include <torch/torch.h>

namespace slamtb {

/**
 * Raises a runtime error with the file, line and message if test is
 * false.
 */
inline void Check(bool test, const char *file, int line, const char *message) {
  if (!test) {
    std::stringstream msg;
    msg << "Check failed: " << file << "(" << line << "): " << message;
    throw std::runtime_error(msg.str());
  }
}

/**
 * Raises a runtime error whatever the test_tensor is in a different
 * device than the espected one.

 * @param expected_dev The expected device.
 * @param test_tensor The tensor which should be tested.
 */
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



}  // namespace slamtb

#define STB_CHECK(test, msg) slamtb::Check(test, __FILE__, __LINE__, msg)
#define STB_CHECK_DEVICE(device, test_tensor) \
  slamtb::CheckDevice(device, test_tensor, __FILE__, __LINE__)
