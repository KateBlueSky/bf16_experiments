#include <iostream>
#include <vector>
#include <chrono>
#include <immintrin.h>
#include <cstring>
#include <cstdint>
#include <cstring>

void convert_f32_to_bf16(const float* src, uint16_t* dst, size_t size) {
    size_t i = 0;

    // Vectorized conversion using AVX512-BF16
#ifdef __AVX512BF16__
    for (; i + 15 < size; i += 16) {
        __m512 f = _mm512_loadu_ps(&src[i]);
        __m256bh bf16 = _mm512_cvtneps_pbh(f);
        _mm256_storeu_si256((__m256i*)(&dst[i]), (__m256i)bf16);
    }
#endif

    // Scalar fallback for remaining values
    for (; i < size; ++i) {
        uint32_t val;
        memcpy(&val, &src[i], sizeof(float));
        dst[i] = static_cast<uint16_t>(val >> 16);
    }
}

int main() {
    const size_t N = 1 << 20; // 1M floats
    std::vector<float> input(N);
    std::vector<uint16_t> output(N);

    // Fill input with dummy data
    for (size_t i = 0; i < N; ++i)
        input[i] = static_cast<float>(i) / 100.0f;

    // Warmup
    convert_f32_to_bf16(input.data(), output.data(), N);

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    convert_f32_to_bf16(input.data(), output.data(), N);
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(end - start).count();
    double bandwidth = (N * (sizeof(float) + sizeof(uint16_t))) / (1e9 * elapsed);

    std::cout << "Conversion time: " << elapsed << " sec\n";
    std::cout << "Throughput: " << bandwidth << " GB/s\n";

    return 0;
}

