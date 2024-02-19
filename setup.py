from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CUDA_FLAGS = []
INSTALL_REQUIREMENTS = []

ext_modules = [
    CUDAExtension('vot.cuda.vot', [
        'vot/cuda/vot_cuda.cpp',
        'vot/cuda/vot_cuda_kernel.cu',
    ]),
]

setup(
    description='PyTorch implementation of "A Voxel Occlusion Tester (VOT)"',
    author='Shun Iwase',
    author_email='siwase@andrew.cmu.edu',
    license='MIT License',
    version='1.0.0',
    name='vot_pytorch',
    packages=['vot', 'vot.cuda'],
    install_requires=INSTALL_REQUIREMENTS,
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension})
