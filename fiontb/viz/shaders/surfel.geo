#version 420

layout(points) in;
layout(triangle_strip, max_vertices=4) out;

// uniform mat4 Modelview;
uniform mat4 ProjModelview;

in Surfel {
  vec4 pos;
  vec3 color;
  vec3 normal;
  float radius;
  float conf;
} gs_in[];

out Frag {
  vec3 color;
  vec3 normal;
  vec2 tc;
  float conf;
} frag;

void main() {

  vec3 pos = gs_in[0].pos.xyz;
  vec3 normal = gs_in[0].normal;
  vec3 u = normalize(vec3(normal.y - normal.z, -normal.x, normal.x));
  vec3 v = vec3(normalize(cross(normal, u)));

  float radius = gs_in[0].radius*.5;
  float aspect = 0.75;
  u *= radius*aspect;
  v *= radius;

  frag.color = gs_in[0].color;
  frag.normal = abs(normal);
  frag.conf = gs_in[0].conf;

  gl_Position = ProjModelview*vec4(pos - u - v, 1.0f);
  frag.tc = vec2(0, 0);
  EmitVertex();

  gl_Position = ProjModelview*vec4(pos + u - v, 1.0);
  frag.tc = vec2(1, 0);
  EmitVertex();

  gl_Position = ProjModelview*vec4(pos - u + v, 1.0);
  frag.tc = vec2(0, 1);
  EmitVertex();
  
  gl_Position = ProjModelview*vec4(pos + u + v, 1.0);
  frag.tc = vec2(1, 1);
  EmitVertex();	
  EndPrimitive();	
}
