#include <torch/torch.h>

#include <tuple>

// CUDA forward declarations

at::Tensor run_cuda(
        const at::Tensor& faces,
        const at::Tensor& mask,
        const at::Tensor& depth_map,
        const int image_height,
        const int image_width);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor run_cpu(
        const at::Tensor& faces,
        const at::Tensor& mask,
        const at::Tensor& depth_map,
        const int image_height,
        const int image_width) {

    CHECK_INPUT(faces);
    CHECK_INPUT(mask);
    CHECK_INPUT(depth_map);

    return run_cuda(faces, mask, depth_map, image_height, image_width);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run_cpu, "Run a voxel occlusion tester");
}
