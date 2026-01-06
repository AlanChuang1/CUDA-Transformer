#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void mul_weight_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ y,
    int64_t n,
    int hidden
) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = x[i] * w[i % hidden];
  }
}

torch::Tensor rmsnorm_fwd_cuda(torch::Tensor x, torch::Tensor weight, double eps) {
  (void)eps; // unused for stub
  auto y = torch::empty_like(x);
  int hidden = (int)x.size(-1);
  int64_t n = x.numel();

  const int threads = 256;
  const int blocks = (int)((n + threads - 1) / threads);
  mul_weight_kernel<<<blocks, threads>>>(
      (const float*)x.data_ptr<float>(),
      (const float*)weight.data_ptr<float>(),
      (float*)y.data_ptr<float>(),
      n, hidden
  );
  return y;
}
