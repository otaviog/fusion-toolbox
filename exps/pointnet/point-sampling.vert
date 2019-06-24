#version 420

in vec4 in_position;

uniform mat4 Modelview;
uniform mat4 ProjModelview;

out Point {
  vec3 position;
} point;

void main() {
  gl_Position = ProjModelview*in_position;
  point.position = (Modelview*in_position).xyz;
}
