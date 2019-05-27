#version 420

layout (location = 0) in vec4 v_point;
layout (location = 1) in int v_index;

uniform mat4 ProjModelview;

out ModelPoint {
  vec3 point;
  flat int index;
} mpoint;

void main() {
  gl_Position = ProjModelview*v_point;

  mpoint.point = (ProjModelview*v_point).xyz;
  mpoint.index = v_index;
}
