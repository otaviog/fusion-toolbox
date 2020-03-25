#version 420

layout (location = 0) in vec4 in_pos;
layout (location = 1) in vec3 in_color;
layout (location = 2) in vec3 in_normal;
layout (location = 3) in float in_radius;
layout (location = 4) in float in_conf;
layout (location = 5) in int in_time;
layout (location = 6) in int in_mask;

uniform float StableThresh;

out Surfel {
  vec4 pos;
  vec3 color;
  vec3 normal;
  float radius;
  float conf;
  flat int time;
  flat int id;	
} vs_out;

void main() {
  if (in_mask == 1) {
	vs_out.time = -1;
	gl_Position = vec4(-10000, -10000, 10000, 0.0);
	return ;
  }
  
  vs_out.pos = in_pos;
  vs_out.color = in_color;
  vs_out.normal = in_normal;
  vs_out.radius = in_radius;
  vs_out.conf = in_conf;
  vs_out.time = in_time;
  vs_out.id = gl_VertexID;
 
  if (StableThresh > 0) {
	if (vs_out.conf < StableThresh) {
	  vs_out.time = -1;
	}
  }
}
