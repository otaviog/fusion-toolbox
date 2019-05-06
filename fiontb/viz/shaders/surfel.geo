#version 420

layout(points) in;
layout(triangle_strip, max_vertices=4) out;

uniform mat4 ProjModelview;

in Surfel {
  vec4 pos;
  vec3 color;
  vec3 normal;
  float radius;
} gs_in[];

out vec3 vert_color;
out vec2 vert_tc;

void main() {

  vec3 pos = gs_in[0].pos.xyz;
  vec3 normal = gs_in[0].normal;
  vec3 u = normalize(vec3(normal.y - normal.z, -normal.x, normal.x));
  vec3 v = vec3(normalize(cross(normal, u)));

  float aspect = 0.75;
  // u *= 0.01;
  //v *= 0.01;
  u *= gs_in[0].radius*aspect;
  v *= gs_in[0].radius;

  gl_Position = ProjModelview*vec4(pos - u - v, 1.0f);
  vert_tc = vec2(0, 0);
  vert_color = gs_in[0].color;
  EmitVertex();

  gl_Position = ProjModelview*vec4(pos + u - v, 1.0);
  vert_tc = vec2(1, 0);
  vert_color = gs_in[0].color;
  EmitVertex();

  gl_Position = ProjModelview*vec4(pos - u + v, 1.0);
  vert_tc = vec2(0, 1);
  vert_color = gs_in[0].color;
  EmitVertex();
  
  gl_Position = ProjModelview*vec4(pos + u + v, 1.0);
  vert_tc = vec2(1, 1);
  vert_color = gs_in[0].color;
  EmitVertex();	
  EndPrimitive();	
}
