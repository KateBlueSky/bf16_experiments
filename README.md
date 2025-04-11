# bf16_experiments

# oneDNN_test.cpp
Compares the performance of the bf16 gemm operations in oneDNN vs oneMKL. As part of the workload converts a float32 array to bfloat16 
To compile to use intrinsics for the float32->bfloat16 conversion
icpx -O2 -mavx512f -mavx512bw -mamx-tile -mamx-bf16 -std=c++17 -I${MKLROOT}/include -L${MKLROOT}/lib -ldnnl -lmkl_intel_lp64 -lmkl_core -march=native -lmkl_sequential -lpthread -ldl DNNTest.cpp -o test_gemm -mavx512bf16 -DDO_NOT_USE_REORDER
To compile to use oneDNN reorder for the float32->bfloat16 conversion
icpx -O2 -mavx512f -mavx512bw -mamx-tile -mamx-bf16 -std=c++17 -I${MKLROOT}/include -L${MKLROOT}/lib -ldnnl -lmkl_intel_lp64 -lmkl_core -march=native -lmkl_sequential -lpthread -ldl DNNTest.cpp -o test_gemm -mavx512bf16

# bf16_conversion_test.cpp
Measure the performance of float32 to bfloat16 conversion performance.
icpx -O2 -mavx512f -mavx512bw -mamx-tile -mamx-bf16 -std=c++17 -lpthread -ldl bf16_conversion_test.cpp -o bf16_conversion_test

Flags
-mamx-bf16 - Enables AMX tile instructions for bfloat16 (BF16).
-mamx-tile - Enables Intel AMX tile instructions.
-mavx512bw - Enables AVX-512 Byte and Word instructions.
-mavx512f - Enables AVX-512 Foundation instructions.
