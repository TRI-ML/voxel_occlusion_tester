import torch as th
import torch.nn as nn

import vot
import vot.cuda.vot as voxel_occlusion_tester_cuda


class VoxelOcclusionTester(nn.Module):
    def __init__(self,
                 image_height=256,
                 image_width=256,
                 grid_size=20.0,
                 K=None):
        super(VoxelOcclusionTester, self).__init__()
        # rendering
        self.image_height = image_height
        self.image_width = image_width
        self.grid_size = grid_size
        face_offset = th.tensor(
                [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [0, 4, 5], [0, 5, 1],
                 [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [4, 0, 3], [4, 3, 7]])
        self.register_buffer('K', K)
        self.register_buffer('face_offset', face_offset)

    def forward(self, all_voxel_centers, mask, depth_map):
        '''
        args:
            all_voxel_centers: (B, N, 3)
            mask: (B, H, W)
            depth_map: (B, H, W)
        '''
        K = self.K
        image_height = self.image_height
        image_width = self.image_width
        grid_size = self.grid_size
        bs, num_voxels = all_voxel_centers.shape[:2]
        device = all_voxel_centers.device

        grid_offset = th.tensor([-0.5 * grid_size, 0.5 * grid_size], device=device)
        grid_pts = th.stack(th.meshgrid(grid_offset, grid_offset, grid_offset, indexing='ij'), dim=-1).reshape(-1, 3)
        vertices = (all_voxel_centers.unsqueeze(-2) + grid_pts[None, None]).reshape(bs, -1, 3)
        vertices = vot.projection(vertices, K, image_height, image_width)
        faces = self.face_offset.repeat(num_voxels, 1) + th.arange(num_voxels).repeat_interleave(12).unsqueeze(-1) * 8
        face_vertices = vot.vertices_to_faces(vertices, faces)
        occ_flag_map = voxel_occlusion_tester_cuda.forward(face_vertices, depth_map, mask, self.image_height, self.image_width)
        print(occ_flag_map)
        return occ_flag_map
