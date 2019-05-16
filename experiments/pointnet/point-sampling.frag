#version 420

in Point {
  vec3 position;
} point;

layout (location=0) out vec3 frag_pos;
layout (location=1) out int frag_selected;
layout (location=2) out vec3 frag_rgb;

void main() {
  frag_pos = point.position;

  float rand = (noise1(123452.2452) + 1.0)/2.0;  
  frag_selected = 1;

  frag_rgb = vec3(1.0, 1.0, 1.0);
}
