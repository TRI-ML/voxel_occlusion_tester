import torch as th

from vot import VoxelOcclusionTester

if __name__ == '__main__':
    K = th.asarray([[572.41136339, 0., 325.2611084],
                         [0., 573.57043286, 242.04899588],
                         [0., 0., 1.]])
    vot = VoxelOcclusionTester(480, 640, 20.0, K)
    print(vot)
