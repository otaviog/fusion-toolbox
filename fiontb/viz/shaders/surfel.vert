#version 420

layout (location = 0) in vec4 vertex_pos;
layout (location = 1) in vec3 vertex_color;
layout (location = 2) in vec3 vertex_normal;
layout (location = 3) in float vertex_radius;

out Surfel {
  vec4 pos;
  vec3 color;
  vec3 normal;
  float radius;
} vs_out;

void main() {
  vs_out.pos = vertex_pos;
  vs_out.color = vertex_color;
  vs_out.normal = vertex_normal;
  vs_out.radius = vertex_radius*0.5;
}
