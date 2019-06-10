#version 420

in flat int frag_map_index;
in flat int frag_frame_index;
in vec3 frag_debug;

layout(location = 0) out int out_map_index;
layout(location = 1) out int out_frame_index;
layout(location = 2) out vec3 out_debug;

void main() {
  out_map_index = frag_map_index + 1;
  out_frame_index = frag_frame_index;
  out_debug = frag_debug;
}
