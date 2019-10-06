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
  flat int id;
}
frag_in;

out vec4 frag_color;

int hash32shiftmult(int key) {
  int c2 = 0x27d4eb2d;
  key = (key ^ 61) ^ (key >> 16);
  key = key + (key << 3);
  key = key ^ (key >> 4);
  key = key * c2;
  key = key ^ (key >> 15);
  return key;
}

void main() {
  if (dot(frag_in.tc, frag_in.tc) > 1.0) {
    discard;
  }

  if (RenderMode == 0) {  // Color
    frag_color.xyz = frag_in.color;
  } else if (RenderMode == 1) {  // Confidence
    frag_color.xyz = texture(ColorMap, vec2(frag_in.conf / MaxConf, 0)).xyz;
  } else if (RenderMode == 2) {  // Normal
    frag_color.xyz = frag_in.normal;
  } else if (RenderMode == 3) {  // No color
    float intensity =
        (frag_in.normal.x + frag_in.normal.y + frag_in.normal.z) / 3.0;
    frag_color.xyz = vec3(intensity, intensity, intensity);
  } else if (RenderMode == 4) {
    frag_color.xyz =
        texture(ColorMap, vec2(float(frag_in.time) / float(MaxTime), 0)).xyz;
	//frag_color.xyz = vec3(float(MaxTime)/2, 0, 0);
  } else if (RenderMode == 5) {
    int id_hash = hash32shiftmult(frag_in.id);

    frag_color.xyz = vec3(((id_hash & 0x00ff0000) >> 16) / 255.0,
                          ((id_hash & 0x0000ff00) >> 8) / 255.0,
                          (id_hash & 0x000000ff) / 255.0);
  }
}
