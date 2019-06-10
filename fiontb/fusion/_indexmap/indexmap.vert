#version 420

layout (location = 0) in vec4 in_point;
layout (location = 1) in vec3 in_normal;

uniform mat4 Modelview;
uniform mat3 NormalModelview;
uniform mat4 ProjModelview;

out ModelPoint {
  vec3 point, normal;
  flat int index;
} mpoint;

void main() {
  gl_Position = ProjModelview*in_point;

  mpoint.point = (Modelview*in_point).xyz;
  mpoint.normal = (NormalModelview*in_normal);
  mpoint.index = gl_VertexID;
}
