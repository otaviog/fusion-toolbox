#version 420 core

uniform int RenderMode;
uniform sampler2D ColorMap;

uniform float MaxConf;
uniform float MaxTime;

in Frag {
  vec3 color;
  vec3 normal;
  vec2 tc;
  float conf;
  flat int time;
} frag_in;

out vec4 frag_color;


void main() {
  if (dot(frag_in.tc, frag_in.tc) > 1.0) {
	discard;
  }

  if (RenderMode == 0) { // Color
	frag_color.xyz = frag_in.color;
  } else if (RenderMode == 1) { // Confidence
	frag_color.xyz = texture(ColorMap, vec2(frag_in.conf/MaxConf, 0)).xyz;
  } else if (RenderMode == 2) { // Normal
	frag_color.xyz = frag_in.normal;
  } else if (RenderMode == 3) { // No color
	float intensity = (frag_in.normal.x + frag_in.normal.y + frag_in.normal.z)/3.0;
	frag_color.xyz = vec3(intensity, intensity, intensity);
  } else if (RenderMode == 4) {
	frag_color.xyz = texture(ColorMap, vec2(float(frag_in.time)/float(MaxTime), 0)).xyz;
  }
}
