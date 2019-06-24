#version 420

layout (location = 0) in vec4 in_pos;

uniform mat4 ModelviewMatrix;
uniform mat4 ProjModelviewMatrix;

out Frag {
  vec3 cam_pos;
  vec3 obj_pos;
} frag;

void main() {
  frag.cam_pos = (ModelviewMatrix*in_pos).xyz;
  frag.obj_pos = in_pos.xyz;
  gl_Position = ProjModelviewMatrix*in_pos;
}
