#version 420

layout (location = 0) in vec4 in_point;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec3 in_mask;

uniform sampler2DRect IndexMapPointsTex;
uniform sampler2DRect IndexMapNormalsTex;
uniform isampler2DRect IndexMapTex;

uniform int ImageWidth;
uniform int ImageHeight;
uniform float Scale;
uniform float MaxNormalAngle;

uniform mat4 ProjModelview;

out flat int frag_map_index;
out flat int frag_frame_index;
out vec3 frag_debug;

float angleBetween(vec3 a, vec3 b) {
  return acos(dot(a, b) / (length(a) * length(b)));
}

void main() {
  if (in_mask == 1) {
	gl_Position = vec4(-10000, -10000, 10000, -10);
	return ;
  }

  float bestDist = 1000;
  float windowMultiplier = 2;

  vec3 ray = vec3(in_point.x, in_point.y, 1);
  float lambda = sqrt(ray.x*ray.x + ray.y*ray.y + 1);

  gl_Position = ProjModelview*in_point;
  float tx = ((gl_Position.x / gl_Position.w) + 1.0)*0.5;
  float ty = ((gl_Position.y / gl_Position.w) + 1.0)*0.5;

  int sx = int(tx*ImageWidth*Scale);
  int sy = int(ty*ImageHeight*Scale);

  int search_size = int(Scale*windowMultiplier);

  int xstart = max(sx - search_size, 0);
  int xend = min(sx + search_size, int(ImageWidth*Scale) - 1);

  int ystart = max(sy - search_size, 0);
  int yend = min(sy + search_size, int(ImageHeight*Scale) - 1);

  int best = 0;
  int found = 0;

  for(int i = xstart; i < xend; i++) {
	for(int j = ystart; j < yend; j++) {
	  ivec2 indexmap = texture(IndexMapTex, vec2(i, j)).xy;

	  if(indexmap.y > 0) {
		int current = indexmap.x;
		
		vec3 vert = texture(IndexMapPointsTex, vec2(i, j)).xyz;
		if(abs((vert.z * lambda) - (in_point.z * lambda)) < 0.05) {
		  float dist = length(cross(ray, vert)) / length(ray);
		  vec3 normal = texture(IndexMapNormalsTex, vec2(i, j)).xyz;

		  if(dist < bestDist
			 && (abs(normal.z) < 0.75f
				 || abs(angleBetween(normal.xyz, in_normal)) < MaxNormalAngle)) {
			found = 1;
			bestDist = dist;
			best = current;
		  }
		}
	  }
	} // for j
  } // for i

  frag_frame_index = gl_VertexID;
  if(found == 1) {
	frag_map_index = best;
	frag_debug = vec3(0, 1, 0);
  } else {
	frag_map_index = 0;
	frag_debug = vec3(1, 0, 0);
  }
}
