from pathlib import Path

import numpy as np
import torch

import tenviz
from fiontb.spatial import DenseVolume

_SHADER_DIR = Path(__file__).parent / "shaders"


class SDFRender(tenviz.DrawProgram):
    @staticmethod
    def _create_texture_space_matrix(verts):
        box_min = verts.min(0)[0]
        diag_size = verts.max(0)[0] - box_min

        texture_space_mtx = torch.eye(4)
        texture_space_mtx[0, 0] = 1.0/diag_size[0]
        texture_space_mtx[1, 1] = 1.0/diag_size[1]
        texture_space_mtx[2, 2] = 1.0/diag_size[2]

        trans = torch.eye(4)

        trans[0, 3] = -box_min[0]
        trans[1, 3] = -box_min[1]
        trans[2, 3] = -box_min[2]

        return texture_space_mtx @ trans

    def __init__(self, dense_volume):
        program = tenviz.load_program_fs(_SHADER_DIR / "sdf.vert",
                                         _SHADER_DIR / "sdf.frag")

        super(SDFRender, self).__init__(tenviz.DrawMode.Quads,
                                        program=program, ignore_missing=True)
        self.volume = dense_volume

        verts = torch.tensor([[-1, 1, 1],  # 0 ltf
                              [-1, -1, 1],  # 1 lbf
                              [1, -1, 1],  # 2 rbf
                              [1, 1, 1],  # 3 rtf
                              [-1, 1, -1],  # 4 ltb
                              [-1, -1, -1],  # 5 lbb
                              [1, -1, -1],  # 6 rbb
                              [1, 1, -1]], dtype=torch.float)  # 7 rtb

        faces = torch.tensor([[0, 3, 2, 1],
                              [4, 5, 6, 7],
                              [0, 4, 5, 1],
                              [3, 2, 6, 7],
                              [0, 3, 7, 4],
                              [1, 5, 6, 2]], dtype=torch.int32)

        self['in_pos'] = verts
        self['NormalModelviewMatrix'] = tenviz.MatPlaceholder.NormalModelview
        self['ProjModelviewMatrix'] = tenviz.MatPlaceholder.ProjectionModelview
        
        self['TextureSpaceMatrix'] = SDFRender._create_texture_space_matrix(
            verts)

        self['ModelviewMatrix'] = tenviz.MatPlaceholder.Modelview
        self['NumTraceStep'] = 10

        self.sdf_3dtex = tenviz.tex_from_torch(
            dense_volume.sdf, tenviz.GLTexTarget.k3D)

        self['SDFSampler'] = self.sdf_3dtex

        self.indices.from_tensor(faces)


def _main():
    volume = DenseVolume(128, 0.01, np.array([0, 0, 0]))
    # volume.sdf = -torch.ones_like(volume.sdf)
    volume.sdf[32:64, 32:64, 32:64] = torch.rand((32, 32, 32))

    ctx = tenviz.Context()
    with ctx.current():
        sdf_render = SDFRender(volume)

    viewer = ctx.viewer([sdf_render])

    while True:
        key = viewer.wait_key(1)
        if key < 0:
            break


if __name__ == '__main__':
    _main()
