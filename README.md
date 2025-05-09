# 🧪 `bf16_experiments`

Experiments with `bfloat16` (BF16) GEMM operations and conversion performance using Intel oneDNN, oneMKL, and AMX/AVX-512 intrinsics.

---

## 📄 `oneDNN_test.cpp`

Benchmark and compare BF16 GEMM performance between **oneDNN** and **oneMKL**.

- Converts `float32` arrays to `bfloat16` as part of the workload.
- Supports both:
  - Intrinsics-based conversion
  - oneDNN's built-in `reorder()` operation

### 🔧 Build Instructions

#### 🔹 Compile with intrinsics for `float32 → bfloat16` conversion:

```bash
icpx -O2 -std=c++17 \
  -mavx512f -mavx512bw -mamx-tile -mamx-bf16 -mavx512bf16 -march=native \
  -I${MKLROOT}/include -L${MKLROOT}/lib \
  DNNTest.cpp -o test_gemm \
  -ldnnl -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -ldl \
  -DDO_NOT_USE_REORDER
```

#### 🔹 Compile with oneDNN's `reorder()` for conversion:

```bash
icpx -O2 -std=c++17 \
  -mavx512f -mavx512bw -mamx-tile -mamx-bf16 -mavx512bf16 -march=native \
  -I${MKLROOT}/include -L${MKLROOT}/lib \
  DNNTest.cpp -o test_gemm \
  -ldnnl -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -ldl
```

---

## 📄 `bf16_conversion_test.cpp`

Benchmark the raw performance of `float32 → bfloat16` conversion using intrinsics (e.g., `vcvtneps2bf16`).

### 🔧 Compile:

```bash
icpx -O2 -std=c++17 \
  -mavx512f -mavx512bw -mamx-tile -mamx-bf16 \
  bf16_conversion_test.cpp -o bf16_conversion_test \
  -lpthread -ldl
```

---

## 📄 `test_mkl_bf16.cpp`

Benchmark the `float32 → bfloat16` conversion using intrinsics + cblas_gemm_bf16bf16f32 vs a single cblas_sgemm call.
The run_test.sh can be used to test a range of values for M N K

### 🔧 Compile:

```bash
icpx -O2 -std=c++17   -mavx512f -mavx512bw -mamx-tile -mamx-bf16 -mavx512bf16 -march=native \
-I${MKLROOT}/include -L${MKLROOT}/lib   test_mkl_bf16.cpp -o test_gemm   \
-lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -ldl
```
---


## 🚩 Compiler Flags Explained

| Flag              | Description |
|-------------------|-------------|
| `-mavx512f`       | Enables AVX-512 Foundation (base 512-bit SIMD instructions). |
| `-mavx512bw`      | Enables AVX-512 Byte/Word operations (8-bit and 16-bit SIMD). |
| `-mamx-tile`      | Enables **AMX tile** instructions (matrix math accelerator). |
| `-mamx-bf16`      | Enables **AMX BF16** dot-product tile instructions. |
| `-mavx512bf16`    | Enables **AVX-512 BF16** conversion instructions (e.g., `vcvtneps2bf16`). |

---




