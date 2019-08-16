#version 420

layout(points) in;
layout(triangle_strip, max_vertices=4) out;

uniform mat4 ProjModelview;

in Surfel {
  vec4 pos;
  vec3 color;
  vec3 normal;
  float radius;
  float conf;
  flat int time;
  flat int id;
} gs_in[];

out Frag {
  vec3 color;
  vec3 normal;
  vec2 tc;
  float conf;
  flat int time;
  flat int id;
} frag;

void main() {
  if (gs_in[0].time < 0) {
	gl_Position = vec4(-10000, -10000, 10000, 1);
	EmitVertex();
	EndPrimitive();
	return;
  }

  vec3 pos = gs_in[0].pos.xyz;
  vec3 normal = gs_in[0].normal;
  vec3 u = normalize(vec3(normal.y - normal.z, -normal.x, normal.x));
  vec3 v = vec3(normalize(cross(normal, u)));

  float radius = gs_in[0].radius*0.1;
  //float aspect = 0.75;
  float aspect = 1.0;
  u *= radius*aspect;
  v *= radius;

  frag.color = gs_in[0].color;
  frag.normal = abs(normal);
  frag.conf = gs_in[0].conf;
  frag.time = gs_in[0].time;
  frag.id = gs_in[0].id;

  gl_Position = ProjModelview*vec4(pos - u - v, 1.0f);
  frag.tc = vec2(-1, -1);
  EmitVertex();

  gl_Position = ProjModelview*vec4(pos + u - v, 1.0);
  frag.tc = vec2(1, -1);
  EmitVertex();

  gl_Position = ProjModelview*vec4(pos - u + v, 1.0);
  frag.tc = vec2(-1, 1);
  EmitVertex();

  gl_Position = ProjModelview*vec4(pos + u + v, 1.0);
  frag.tc = vec2(-1, -1);
  EmitVertex();
  EndPrimitive();
}
