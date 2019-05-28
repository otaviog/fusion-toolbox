#version 420

layout (location = 0) in vec4 in_pos;
layout (location = 1) in vec3 in_color;
layout (location = 2) in vec3 in_normal;
layout (location = 3) in float in_radius;
layout (location = 4) in float in_conf;

out Surfel {
  vec4 pos;
  vec3 color;
  vec3 normal;
  float radius;
  float conf;
} vs_out;

void main() {
  vs_out.pos = in_pos;
  vs_out.color = in_color;
  vs_out.normal = in_normal;
  vs_out.radius = in_radius*0.5;
  vs_out.conf = in_conf;
}
