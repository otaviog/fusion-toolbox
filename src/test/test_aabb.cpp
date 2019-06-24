#include "catch.hpp"

#include <fiontb/aabb.hpp>

TEST_CASE("AABB", "[aabb]") {
  fiontb::AABB box2(Eigen::Vector3f(2, 1, 3), Eigen::Vector3f(5, 4, 9));

  // Left half intersection
  CHECK(box2.Intersects(Eigen::Vector3f(1, 3, 6), Eigen::Vector3f(2.5, 1.5, 4),
                        Eigen::Vector3f(3.5, 3.5, 6)));

  // Left no intersection
  CHECK(!box2.Intersects(Eigen::Vector3f(1, 3, 6), Eigen::Vector3f(1.8, 1.5, 4),
                         Eigen::Vector3f(1.9, 3.5, 6)));

  // Right half intersection
  CHECK(box2.Intersects(Eigen::Vector3f(4, 3, 6), Eigen::Vector3f(5.5, 1.5, 4),
                        Eigen::Vector3f(7.5, 3.5, 6)));

  fiontb::AABB box3(Eigen::Vector3f(-0.0931465998, 0.107758701, 0.000578399748),
                    Eigen::Vector3f(-0.017493749, 0.181896999, 0.0578007996));
  CHECK(
      box3.IsInside(Eigen::Vector3f(-0.0235083941, 0.126511514, 0.0114857228)));

  Eigen::Vector3f p0(-0.00632139016, 0.0931651965, -0.0328935012);
  Eigen::Vector3f p1(-0.0144477999, 0.0879274011, -0.0380297005);
  Eigen::Vector3f p2(-0.0184118003, 0.0948058963, -0.0307433996);

  
}
