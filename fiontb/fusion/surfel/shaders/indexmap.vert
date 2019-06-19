#version 420

layout (location = 0) in vec4 in_point;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in int in_mask;

uniform mat4 Modelview;
uniform mat3 NormalModelview;
uniform mat4 ProjModelview;

out ModelPoint {
  vec3 point, normal;
  flat int index;
} mpoint;

void main() {
  if (in_mask == 1) {
	gl_Position = vec4(-10000, -10000, 10000, 1);
	return ;
  }
  gl_Position = ProjModelview*in_point;
  mpoint.point = (Modelview*in_point).xyz;
  mpoint.normal = NormalModelview*in_normal;
  mpoint.index = gl_VertexID;
}
