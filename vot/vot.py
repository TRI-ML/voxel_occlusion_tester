import torch as th
import torch.nn as nn

import vot
import vot.cuda.vot as voxel_occlusion_tester_cuda


class VoxelOcclusionTester(nn.Module):
    def __init__(self,
                 image_height=256,
                 image_width=256,
                 grid_size=20.0):
        super(VoxelOcclusionTester, self).__init__()
        # rendering
        self.image_height = image_height
        self.image_width = image_width
        grid_offset = grid_size * (th.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]) - 0.5)
        face_offset = th.tensor(
                [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [0, 4, 5], [0, 5, 1],
                 [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [4, 0, 3], [4, 3, 7]])
        # self.register_buffer('K', K)
        self.register_buffer('grid_offset', grid_offset)
        self.register_buffer('face_offset', face_offset)

    def forward(self, all_voxel_centers, mask, depth_map, K):
        '''
        args:
            all_voxel_centers: (B, N, 3)
            mask: (B, H, W)
            depth_map: (B, H, W)
            K: (B, 3, 3)
        '''
        bs, num_voxels = all_voxel_centers.shape[:2]
        device = all_voxel_centers.device

        # K = self.K.unsqueeze(0).repeat(bs, 1, 1)
        image_height = self.image_height
        image_width = self.image_width

        vertices = (all_voxel_centers.unsqueeze(-2) + self.grid_offset[None, None]).reshape(bs, -1, 3)
        vertices = vot.projection(vertices, K, image_height, image_width)
        faces = self.face_offset.repeat(num_voxels, 1) + th.arange(num_voxels, device=device).repeat_interleave(12).unsqueeze(-1) * 8
        faces = faces.unsqueeze(0).repeat(bs, 1, 1)
        face_vertices = vot.vertices_to_faces(vertices, faces)
        occ_flag_map = voxel_occlusion_tester_cuda.run(face_vertices, mask, depth_map, self.image_height, self.image_width)
        return occ_flag_map
