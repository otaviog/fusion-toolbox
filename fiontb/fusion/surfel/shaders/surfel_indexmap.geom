#version 420

layout(points) in;
layout(triangle_strip, max_vertices=4) out;

uniform mat4 ProjModelview;

in Surfel {
  vec4 pos;
  vec4 pos_conf;
  vec4 normal_rad;
  vec3 color;
  flat int index;
  flat int time;
} vert[];

out GeomSurfel {
  vec4 pos_conf;
  vec4 normal_rad;
  vec3 color;
  vec2 tc;
  flat int index;
  flat int time;
} frag;


void main() {
  if (vert[0].index < 0) {
	gl_Position = vec4(-10000, -10000, 10000, -10);
	EmitVertex();
	EndPrimitive();
	return;
  }

  vec3 pos = vert[0].pos.xyz;
  vec3 normal = vert[0].normal_rad.xyz;
  vec3 u = normalize(vec3(normal.y - normal.z, -normal.x, normal.x));
  vec3 v = vec3(normalize(cross(normal, u)));

  float radius = vert[0].normal_rad.w;
  //float radius = .1;
  float aspect = 1.0;
  u *= radius*aspect;
  v *= radius;

  frag.pos_conf = vert[0].pos_conf;
  frag.normal_rad = vert[0].normal_rad;
  frag.color = vert[0].color;
  frag.index = vert[0].index;
  frag.time = vert[0].time;

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
