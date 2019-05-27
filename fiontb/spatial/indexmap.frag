#version 420

in ModelPoint {
  vec3 point;
  flat int index;
} model;

layout(location = 0) out vec4 debug_color;
layout(location = 1) out int mpoint_index;
layout(location = 2) out vec3 mpoint;

void main() {
  debug_color = vec4(255, 0, 0, 1);
  mpoint_index = model.index;  
  mpoint = model.point;
}
