#version 420

layout (location = 0) in vec4 in_point;
layout (location = 1) in vec3 in_normal;

uniform sampler2DRect IndexMapPointsTex;
uniform sampler2DRect IndexMapNormalsTex;
uniform isampler2DRect IndexMapTex;

uniform int ImageWidth;
uniform int ImageHeight;
uniform float Scale;

uniform mat4 ProjModelview;

out flat int frag_map_index;
out flat int frag_frame_index;
out vec3 frag_debug;

float angleBetween(vec3 a, vec3 b) {
    return acos(dot(a, b) / (length(a) * length(b)));
}

void main() {
  uint best = 0U;    
  int found = 0;
	    
  float indexXStep = (1.0f / (ImageWidth * Scale)) * 0.5f;
  float indexYStep = (1.0f / (ImageHeight * Scale)) * 0.5f;
	    
  float bestDist = 1000;
	    
  float windowMultiplier = 2;
        
  vec3 ray = vec3(in_point.x, in_point.y, 1);
  float lambda = sqrt(ray.x*ray.x + ray.y*ray.y + 1);

  gl_Position = ProjModelview*in_point;
  float tx = ((gl_Position.x / gl_Position.w) + 1.0)*0.5;
  float ty = 1.0 - ((gl_Position.y / gl_Position.w) + 1.0)*0.5;

  float some_z = -1000;
  for(float i = tx - (Scale * indexXStep * windowMultiplier);
	  i < tx + (Scale * indexXStep * windowMultiplier);
	  i+=indexXStep) {
	for(float j = ty - (Scale * indexYStep * windowMultiplier);
		j < ty + (Scale * indexYStep * windowMultiplier);
		j+=indexYStep) {
	  float ii = i*ImageWidth;
	  float jj = (1.0 - j)*ImageHeight;
	  int current = int(texture(IndexMapTex, vec2(ii, jj)));
	           
	  if(current > 0) {
		vec3 vert = texture(IndexMapPointsTex, vec2(ii, jj)).xyz;
		if (some_z < -999)
		  some_z = vert.z;
		if(abs((vert.z * lambda) - (in_point.z * lambda)) < 0.05) {
		  float dist = length(cross(ray, vert)) / length(ray);		  
		  vec3 normal = texture(IndexMapNormalsTex, vec2(ii, jj)).xyz;
                       
		  if(dist < bestDist
			 && (abs(normal.z) < 0.75f
				 || abs(angleBetween(normal.xyz, in_normal)) < 0.5f)) {
			found = 1;
			bestDist = dist;
			best = current;
		  }
		}
	  }
	}
  }

  frag_frame_index = gl_VertexID;
  if(found == 1) {
	//frag_map_index = int(best);
	//frag_map_index = int(texture(IndexMapTex, vec2(125, 250)).x);
	frag_map_index = int(texture(IndexMapTex, vec2(tx*ImageWidth, (1.0 - in_point.y)*ImageHeight)).x);
	//frag_map_index = 5;
	frag_debug = vec3(0, 1, 0);
  } else {	
	frag_map_index = 0;
	frag_debug = vec3(1, 0, 0);
  }

  frag_map_index = int(texture(IndexMapTex, vec2(tx*ImageWidth, (1.0 - in_point.y)*ImageHeight)).x);

  //frag_map_index = found;
}
