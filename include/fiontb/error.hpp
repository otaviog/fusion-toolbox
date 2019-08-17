#pragma once

#include <sstream>

namespace fiontb {

inline void Check(bool test, const char *file, int line, const char *message) {
  if (!test) {
    std::stringstream msg;
    msg << "Check failed: " << file << "(" << line << "): " << message;
    throw std::runtime_error(msg.str());
  }
}

#define FTB_CHECK(test, msg) fiontb::Check(test, __FILE__, __LINE__, msg)

}  // namespace fiontb
