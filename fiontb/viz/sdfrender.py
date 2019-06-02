import tenviz


class SDFRender(tenviz.DrawProgram):
    def __init__(self, dense_volume):        
        super(SDFRender, self).__init__(tenviz.DrawMode.Quads)
        self.volume = dense_volume
        
        self.sdf_3dtex = tenviz.tex_from_torch(dense_volume.sdf, tenviz.GLTexTarget.k3D)

        self['SDFSampler'] = self.sdf_3dtex
        
        
        
