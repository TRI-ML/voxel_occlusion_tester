import torch as th

from vot import VoxelOcclusionTester

if __name__ == '__main__':
    K = th.asarray([[572.41136339, 0., 325.2611084],
                         [0., 573.57043286, 242.04899588],
                         [0., 0., 1.]])
    vot = VoxelOcclusionTester(480, 640, 20.0, K)
    vot.cuda()

    all_voxel_centers = th.ones(1, 20000, 3).cuda() * 2000
    all_voxel_centers[:, :, 0] = 0.0
    all_voxel_centers[:, :, 1] = 0.0
    depth_map = th.ones(1, 480, 640).cuda() * 100
    mask = th.ones(1, 480, 640).bool().cuda()

    occ_flag_map = vot(all_voxel_centers, mask, depth_map)

    start = th.cuda.Event(enable_timing=True)
    end = th.cuda.Event(enable_timing=True)
    start.record()
    occ_flag_map = vot(all_voxel_centers, mask, depth_map)
    end.record()
    th.cuda.synchronize()
    print(start.elapsed_time(end))
    print(occ_flag_map)
