#version 420

in Query {
  flat int index;
  vec3 point;
} query;


uniform isampler2DRect IndexSampler;
uniform samplerRect ModelPointSampler;

uniform int IndexYSteps;
uniform int IndexXSteps;

layout(location = 0) out vec4 debug_color;
layout(location = 1) out int query_index;
layout(location = 2) out int model_index;

vec4 color(int index, int max_index) {
  if (index < 0) {
	return vec4(1, 1, 1, 1);
  }
  
  float idx = float(index);
  float mi = float(max_index);
  float range = mi*0.3;
  
  if (index > int(mi*0.6)) {
	return vec4(float(index) / range, 0, 0, 1);
  }

  if (index > mi*0.3) {
	return vec4(0, float(index) / range, 0, 1);
  }

  return vec4(0, 0, float(index) / range, 1);  
}

void main() {
  float best_dist = 100000.0;
  int best_index = -1;
  vec3 qpoint = query.point;
  
  int IndexXSteps = 8;
  int IndexYSteps = 8;
  for (int row=0; row<IndexYSteps; ++row) {
	float yl = int(gl_FragCoord.y + IndexYSteps/2 - row);

	for (int col=0; col<IndexXSteps; ++col) {
	  float xl = int(gl_FragCoord.x - IndexXSteps/2 + col);

	  int current = int(texture(IndexSampler, vec2(xl, yl)).x);
	  //current = 0;
	  if (current > 0) {
		vec3 model_point = texture(ModelPointSampler, vec2(xl, yl)).xyz;
		
		float dist = length(model_point - query.point);		
		if (dist < best_dist) {
		  best_dist = dist;
		  best_index = current;
		}
	  }
	}
  }

  debug_color = color(best_index, 1000);
  query_index = query.index;
  model_index = best_index;
}
