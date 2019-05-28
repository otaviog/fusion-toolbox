#version 420 core

uniform int RenderMode;
uniform sampler2D BaseTexture;
uniform sampler2D ColorMap;

uniform float MaxConf;

in Frag {
  vec3 color;
  vec2 tc;
  float conf;
} frag_in;

out vec4 frag_color;


void main() {
  vec4 tex_color = texture(BaseTexture, frag_in.tc);
  if (tex_color.w < 0.5) {
	discard;
  }

  if (RenderMode == 0) {
	frag_color.xyz = frag_in.color;
  } else if (RenderMode == 1) {
	frag_color.xyz = texture(ColorMap, vec2(frag_in.conf/MaxConf, 0)).xyz;
  }

}
