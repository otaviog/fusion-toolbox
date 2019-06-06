#version 420

uniform sampler3D SDFSampler;
uniform int NumTraceStep;
uniform mat4 ModelviewMatrix;
uniform mat3 NormalModelviewMatrix;
uniform mat4 TextureSpaceMatrix;

in Frag {
  vec3 cam_pos;
  vec3 obj_pos;
} frag;

layout (location=0) out vec4 frag_color;

void main() {  
  int NumTraceStep2 = 400;
  float step_size = 1.0/float(NumTraceStep2);

  vec3 ray_pos = frag.cam_pos.xyz;
  vec3 eye_vec = -normalize(frag.cam_pos.xyz);
	
  float last_sdf = 0.0;
  float col = 0.0;
  bool d = true;
  for (int i=0; i<NumTraceStep2; ++i) {
	vec3 obj_ray_pos = (inverse(ModelviewMatrix)*vec4(ray_pos, 1.0)).xyz;
	vec3 tex_coord = (TextureSpaceMatrix*vec4(obj_ray_pos, 1)).xyz;
	float curr_sdf = texture(SDFSampler, tex_coord).x;
	
	if ((curr_sdf < 0 && last_sdf > 0)
		||(curr_sdf > 0 && last_sdf < 0)) {
	  col = 1.0;
	  d = false;
	}
	last_sdf = curr_sdf;	

	ray_pos = ray_pos - eye_vec*step_size;
  }

  if (d) {
	discard;
  }
  
  frag_color = vec4(col, 0, 0, 1);
}
