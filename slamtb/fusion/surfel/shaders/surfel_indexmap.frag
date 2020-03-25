#version 420

layout(location = 0) out vec4 out_point_conf;
layout(location = 1) out vec4 out_normal_rad;
layout(location = 2) out vec3 out_color;
layout(location = 3) out ivec3 out_index;

in GeomSurfel {
  vec4 pos_conf;
  vec4 normal_rad;
  vec3 color;
  vec2 tc;
  flat int index;
  flat int time;
} surfel;

void main() {
  if (dot(surfel.tc, surfel.tc) > 1.0) {
    discard;
  }
  out_point_conf = surfel.pos_conf;
  out_normal_rad = surfel.normal_rad;
	
  out_color = surfel.color;
	
  out_index.x = surfel.index;
  out_index.y = 1;
  out_index.z = surfel.time;
}
