#include <torch/extension.h>

torch::Tensor rmsnorm_fwd_cuda(torch::Tensor x, torch::Tensor weight, double eps);

torch::Tensor rmsnorm_fwd(torch::Tensor x, torch::Tensor weight, double eps) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  return rmsnorm_fwd_cuda(x, weight, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rmsnorm_fwd", &rmsnorm_fwd, "RMSNorm forward (CUDA)");
}
