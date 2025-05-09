#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <dnnl.hpp>
#include <mkl.h>
#include <immintrin.h>
#include <cstring>

//#define DO_NOT_USE_REORDER


using namespace dnnl;

void convert_bf16_to_f32(const uint16_t* src, float* dst, size_t size) {
    size_t i = 0;

#ifdef __AVX512F__
    
    //std::cout << "using __AVX512F__\n";
    for (; i + 15 < size; i += 16) {
        __m256i bf16_vals = _mm256_loadu_si256((const __m256i*)&src[i]);

        // Zero extend to 32 bits by shifting left 16 bits
        __m512i expanded = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_vals), 16);

        // Bitcast to float
        __m512 f32_vals = _mm512_castsi512_ps(expanded);

        _mm512_storeu_ps(&dst[i], f32_vals);
    }
#endif

    // Scalar fallback
    for (; i < size; ++i) {
        uint32_t val = static_cast<uint32_t>(src[i]) << 16;
        std::memcpy(&dst[i], &val, sizeof(float));
    }
}

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





// ---- GEMM Config ----
//constexpr int M = 256;
//constexpr int M = 512;
//constexpr int K = 512;
//constexpr int N = 20;
//constexpr int K = 256;
//constexpr int N = 256;

int main(int argc, char* argv[]) {
    //std::cout << "Benchmarking GEMM: " << M << "x" << K << " x " << K << "x" << N << "\n";

    int M = std::atoi(argv[1]);
    int K = std::atoi(argv[2]);
    int N = std::atoi(argv[3]);

    // Allocate matrices
    std::vector<float> A(M * K), B(K * N), C_mkl(M * N), C_dnnl(M * N);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& x : A) x = dist(gen);
    for (auto& x : B) x = dist(gen);

    // === MKL GEMM (float32) ===
    {
        std::fill(C_mkl.begin(), C_mkl.end(), 0.0f);
        

	mkl_set_num_threads_local(1);

        for (int i = 0; i < 5; i++)
	{

        
        //std::vector<uint16_t> A_bf(M * K), B_bf(K * N);
        auto start_f32 = std::chrono::high_resolution_clock::now();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,          
                    M, N, K, 
                    1.0f,
                    A.data(), K,
                    B.data(), N,
                    0.0f,
                    C_mkl.data(), N);

        auto end_f32 = std::chrono::high_resolution_clock::now();
        auto elapsed_f32 = std::chrono::duration_cast<std::chrono::microseconds>(end_f32 - start_f32);
        //std::cout << "MKL float32 GEMM time: " << elapsed_f32 << " sec\n"; 

        auto start = std::chrono::high_resolution_clock::now();
        
        std::vector<uint16_t> A_bf(M * K), B_bf(K * N);
        convert_f32_to_bf16(A.data(), A_bf.data(), M * K);
        convert_f32_to_bf16(B.data(), B_bf.data(), K * N);
  

        //auto start = std::chrono::high_resolution_clock::now();

        //cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasNoTrans,          
                    M, N, K, 
                    1.0f,
                    A_bf.data(), K,
                    B_bf.data(), N,
                    0.0f,
                    C_mkl.data(), N);

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_bf16 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << M << "," << K << "," << N << "," << elapsed_bf16.count() << "," << elapsed_f32.count() << std::endl; 
	
	//std::cout << "MKL float16 GEMM time: " << elapsed << " sec\n";
	}
    }
}
