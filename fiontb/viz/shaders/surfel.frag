#version 420 core

uniform sampler2D BaseTexture;

in vec3 vert_color;
in vec2 vert_tc;

out vec4 frag_color;


void main() {
  frag_color = texture(BaseTexture, vert_tc);
  frag_color.xyz = vert_color;
  if (frag_color.w < 0.4) {
	discard;
  }
}
