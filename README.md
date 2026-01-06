# CUDA-Accelerated Transformer Inference Ops (PyTorch Extension)

## TL;DR
Built a **PyTorch C++/CUDA extension** that accelerates transformer inference by implementing a **fused RMSNorm forward kernel** optimized for **low-latency decoding**.

Achieved **X% lower kernel time** and **Y% lower end-to-end p95 latency** in a **FastAPI inference microservice benchmark** compared to the PyTorch baseline.

---

## Why This Exists
Transformer inference spends a non-trivial amount of time in **small, bandwidth-bound operations** (normalization, softmax, sampling).  
At low batch sizes—common during decoding—Python overhead and non-fused kernels significantly increase tail latency.

This project focuses on one high-impact operation:

**Fused RMSNorm (CUDA)**  
Computes:
```
y = x * rsqrt(mean(x^2) + eps) * weight
```
in a **single CUDA kernel**, minimizing memory traffic and kernel launch overhead.

---

## Features
- ✅ PyTorch extension via `torch.utils.cpp_extension`
- ✅ Custom CUDA kernel with profiling-driven optimizations
- ✅ Unit tests against PyTorch reference implementation
- ✅ Benchmark harness reporting **p50 / p95 / p99 latency** and throughput
- ✅ Docker environment for reproducible builds

---

## Results
**Hardware:** A100 / RTX 4090 / T4 *(fill in your GPU)*

### Microbench (Kernel Time)
| Op      | Shape (B, T, H) | PyTorch (µs) | CUDA Fused (µs) | Speedup |
|--------|------------------|--------------|------------------|---------|
| RMSNorm | (1, 1, 4096)     | XX           | YY               | Z.Zx    |
| RMSNorm | (4, 1, 4096)     | XX           | YY               | Z.Zx    |
| RMSNorm | (8, 1, 4096)     | XX           | YY               | Z.Zx    |

### End-to-End Decode-Step Latency  
**FastAPI server, concurrency = N**

| Metric | Baseline | CUDA RMSNorm | Improvement |
|------|----------|--------------|-------------|
| p50  | XX ms    | YY ms        | -Z%         |
| p95  | XX ms    | YY ms        | -Z%         |
| p99  | XX ms    | YY ms        | -Z%         |

---

## How to Run

### 1. Create Environment
```bash
conda create -n cuda-ext python=3.10 -y
conda activate cuda-ext
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Build Extension
```bash
pip install -e .
```

### 4. Run Tests
```bash
pytest -q
```

### 5. Run Benchmarks
```bash
python bench/bench_rmsnorm.py --device cuda --hidden 4096 --batch 1 --tokens 1
python bench/bench_server.py --concurrency 32 --duration 60
```

---

## Profiling
- **Nsight Systems**: end-to-end tracing and kernel launch analysis
- **Nsight Compute**: kernel-level metrics (occupancy, memory throughput, warp efficiency)

---

## Repository Layout
```text
ext/        # C++ bindings and CUDA kernel implementation
bench/      # Microbenchmarks and FastAPI server benchmark
tests/      # Correctness tests vs PyTorch reference
docker/     # Reproducible CUDA build environment
```

---

## Notes on Optimization
- Single-pass fused kernel to minimize global memory traffic
- Vectorized loads/stores where applicable
- Warp-level reduction for mean-square computation
- Avoided intermediate tensor allocations and Python overhead

---

## Roadmap
- [ ] Backward kernel + autograd support
- [ ] FP16 / BF16 support
- [ ] CUDA Graphs for stable low-latency execution
- [ ] Fused residual + RMSNorm kernel
