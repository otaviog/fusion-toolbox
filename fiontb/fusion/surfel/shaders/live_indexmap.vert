#version 420

layout (location = 0) in vec4 in_point;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in float in_conf;
layout (location = 3) in float in_radius;

uniform mat4 ProjModelview;
uniform mat4 Modelview;
uniform mat3 NormalModelview;

out Surfel {
  vec3 pos;
  vec4 normal_rad;
  vec3 color;
  flat int index;
} surfel;

void main() {
  gl_Position = ProjModelview*in_point;
  float tx = ((gl_Position.x / gl_Position.w) + 1.0)*0.5;
  float ty = ((gl_Position.y / gl_Position.w) + 1.0)*0.5;

  surfel.pos = (Modelview*in_point).xyz;
  surfel.normal_rad.xyz = NormalModelview*in_normal;
  surfel.normal_rad.w = in_radius;

  surfel.index = gl_VertexID;  
}
