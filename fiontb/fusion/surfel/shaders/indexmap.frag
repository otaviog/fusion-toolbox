#version 420

in Surfel {
  vec4 pos_conf;
  vec4 normal_rad;
  vec3 color;
  flat int index;
} surfel;

layout(location = 0) out vec4 out_point_conf;
layout(location = 1) out vec4 out_normal_rad;
layout(location = 2) out ivec3 out_index;
layout(location = 3) out vec3 out_color;

void main() {
  out_point_conf.xy = surfel.pos_conf.xy;
  out_point_conf.z = -surfel.pos_conf.z;
  out_point_conf.w = surfel.pos_conf.w;

  out_normal_rad = surfel.normal_rad;

  out_color = surfel.color;
	
  out_index.x = surfel.index;
  out_index.y = 1;
}
