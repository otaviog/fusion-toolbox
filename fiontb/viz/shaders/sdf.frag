#version 420

uniform sampler3D SDFSampler;
uniform int NumTraceStep;
uniform mat4 ModelviewMatrix;

in Frag {
  vec3 world_pos;
  vec3 local_pos;
} frag;

layout (location = 0) vec4 frag_color;

void main() {
  float step_size = 1.0/float(NumTraceStep);

  vec3 ray_pos = frag.local_pos;
  vec3 eye_vec = -normalize(frag.local_pos);
  	
  for (int i=0; i<NumTraceStep; ++i) {
	vec3 tex_coord = (TextureSpaceMatrix*ray_pos).xyz;
	vec4 voxel = texture(SDFSampler, tex_coord);
	
	vec3 curr_world_pos = (ModelviewMatrix*ray_pos).xyz;
	ray_pos = ray_pos - eye_vec*step_size;
  }

  
}
