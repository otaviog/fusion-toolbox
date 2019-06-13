#version 420

in ModelPoint {
  vec3 point, normal;
  flat int index;
} mpoint;

layout(location = 0) out vec3 out_point;
layout(location = 1) out vec3 out_normal;
layout(location = 2) out int out_index;

void main() {

  out_point.xy = mpoint.point.xy;
  out_point.z = -mpoint.point.z;
  
  out_normal = mpoint.normal;
  out_index = mpoint.index + 1;
}
