#version 420

in Surfel {
  vec3 pos;
  vec3 normal;
  flat int index;
} surfel;

layout(location = 0) out vec3 out_point;
layout(location = 1) out vec3 out_normal;
layout(location = 2) out ivec3 out_index;

void main() {
  out_point.xy = surfel.pos.xy;
  out_point.z = -surfel.pos.z;

  out_normal = surfel.normal;
  out_index.x = surfel.index + 1;
}
