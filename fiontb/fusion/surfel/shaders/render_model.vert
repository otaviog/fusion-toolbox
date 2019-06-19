#version 420

layout (location = 0) in vec4 in_point;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in float in_conf;
layout (location = 3) in int in_mask;
layout (location = 4) in int in_time;

uniform mat4 ProjModelview;
uniform mat4 Modelview;
uniform mat3 NormalModelview;

uniform float StableThresh;
uniform int Time;

out Surfel {
  vec3 pos;
  vec3 normal;
  flat int index;
} surfel;

void main() {
  if (in_mask == 1
	  || (Time >= 0 && in_time != Time) // Filter new added
	  || (StableThresh > 0.0 && in_conf < StableThresh)) { // Filter non-stable
	gl_Position = vec4(-10000, -10000, 10000, -10);
	return;
  }

  gl_Position = ProjModelview*in_point;
  float tx = ((gl_Position.x / gl_Position.w) + 1.0)*0.5;
  float ty = ((gl_Position.y / gl_Position.w) + 1.0)*0.5;

  surfel.pos = (Modelview*in_point).xyz;
  surfel.normal = NormalModelview*in_normal;
  surfel.index = gl_VertexID;  
}
