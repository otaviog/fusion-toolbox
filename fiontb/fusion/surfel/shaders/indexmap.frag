#version 420

in ModelPoint {
  vec3 point, normal;
  flat int index;
} mpoint;

layout(location = 2) out vec3 out_point;
layout(location = 1) out vec3 out_normal;
layout(location = 0) out ivec3 out_index;
layout(location = 3) out vec3 out_debug;

void main() {

  out_point.xy = mpoint.point.xy;
  out_point.z = -mpoint.point.z;
  
  out_normal = mpoint.normal;
  out_index.x = mpoint.index + 1;

  out_debug = vec3(1, 0, 1);
}
