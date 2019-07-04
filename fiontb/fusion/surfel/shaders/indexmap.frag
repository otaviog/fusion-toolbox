#version 420

in Surfel {
  vec3 pos;
  vec4 normal_rad;
  vec3 color;
  flat int index;
} surfel;

layout(location = 0) out vec3 out_point;
layout(location = 1) out vec4 out_normal_rad;
layout(location = 2) out ivec3 out_index;
layout(location = 3) out vec3 out_color;

void main() {
  out_point.xy = surfel.pos.xy;
  out_point.z = -surfel.pos.z;

  out_normal_rad = surfel.normal_rad;

  out_color = surfel.color;
	
  out_index.x = surfel.index;
  out_index.y = 1;
}
