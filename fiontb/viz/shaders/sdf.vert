#version 420

layout (location = 0) in vec4 in_pos;

uniform mat4 ModelviewMatrix;
uniform mat4 ProjModelviewMatrix;

out Frag {
  vec3 world_pos;
} frag;

void main() {
  gl_Position = ProjModelviewMatrix*inpos;

  frag.world_pos = inpos*ModelviewMatrix
}
