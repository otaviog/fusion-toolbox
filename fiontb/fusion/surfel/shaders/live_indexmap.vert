#version 420

layout (location = 0) in vec4 in_point;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in float in_conf;
layout (location = 3) in float in_radius;

uniform mat4 ProjModelview;
uniform mat4 Modelview;
uniform mat3 NormalModelview;

out Surfel {
  vec4 pos_conf;
  vec4 normal_rad;
  vec3 color;
  flat int index;
  flat int time;
} surfel;

void main() {
  gl_Position = ProjModelview*in_point;

  surfel.pos_conf.xyz = (Modelview*in_point).xyz;
  surfel.normal_rad.xyz = NormalModelview*in_normal;
  surfel.normal_rad.w = in_radius;

  surfel.index = gl_VertexID;
  surfel.time = 0;
}
