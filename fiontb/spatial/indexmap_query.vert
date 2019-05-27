#version 420

layout (location = 0) in vec4 v_qpoint;
layout (location = 1) in int v_qindex;

uniform mat4 ProjModelview;

out Query {
  flat int index;
  vec3 point;
} query;


void main() {
  gl_Position = ProjModelview*v_qpoint;
  query.index = v_qindex;
  query.point = (ProjModelview*v_qpoint).xyz;
}
