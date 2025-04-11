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
constexpr int M = 1024;
//constexpr int K = 512;
constexpr int K = 20;
constexpr int N = 20;

int main() {
    std::cout << "Benchmarking GEMM: " << M << "x" << K << " x " << K << "x" << N << "\n";

    // Allocate matrices
    std::vector<float> A(M * K), B(K * N), C_mkl(M * N), C_dnnl(M * N);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& x : A) x = dist(gen);
    for (auto& x : B) x = dist(gen);

    // === MKL GEMM (float32) ===
    {
        std::fill(C_mkl.begin(), C_mkl.end(), 0.0f);
        
        //std::vector<uint16_t> A_bf(M * K), B_bf(K * N);
        //convert_f32_to_bf16(A.data(), A_bf.data(), M * K);
        //convert_f32_to_bf16(B.data(), B_bf.data(), K * N);
  

        auto start = std::chrono::high_resolution_clock::now();

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        //cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasNoTrans,          
                    M, N, K, 
                    1.0f,
                    A.data(), K,
                    B.data(), N,
                    0.0f,
                    C_mkl.data(), N);

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        std::cout << "MKL float32 GEMM time: " << elapsed << " sec\n";
    }

    // === oneDNN GEMM (bf16) ===
    
    
    #ifdef DO_NOT_USE_REORDER
    {

        std::cout << "Using intrinsics to do the conversion from float32 to bfloat16.\n";
       

        //auto start = std::chrono::high_resolution_clock::now();
        
        engine eng(engine::kind::cpu, 0);
        stream s(eng);

       
        std::vector<uint16_t> A_bf(M * K), B_bf(K * N), C_bf(M * N);
         
        auto start = std::chrono::high_resolution_clock::now();
        convert_f32_to_bf16(A.data(), A_bf.data(), M * K);
        convert_f32_to_bf16(B.data(), B_bf.data(), K * N);

        //engine eng(engine::kind::cpu, 0);
        //stream s(eng);

        memory::dims a_dims = { M, K };
        memory::dims b_dims = { K, N };
        memory::dims c_dims = { M, N };

        auto a_md = memory::desc(a_dims, memory::data_type::bf16, memory::format_tag::ab);
        auto b_md = memory::desc(b_dims, memory::data_type::bf16, memory::format_tag::ab);
        auto c_md = memory::desc(c_dims, memory::data_type::bf16, memory::format_tag::ab);

        auto A_mem = memory(a_md, eng, A_bf.data());
        auto B_mem = memory(b_md, eng, B_bf.data());
        auto C_mem = memory(c_md, eng, C_bf.data());

        auto matmul_pd = matmul::primitive_desc(eng, a_md, b_md, c_md);
        auto matmul_prim = matmul(matmul_pd);

        //auto start = std::chrono::high_resolution_clock::now();

        matmul_prim.execute(s, {
            {DNNL_ARG_SRC, A_mem},
            {DNNL_ARG_WEIGHTS, B_mem},
            {DNNL_ARG_DST, C_mem}
        });


        convert_bf16_to_f32(C_bf.data(), C_dnnl.data(), M * N);
        s.wait();
        

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        std::cout << "oneDNN bf16 GEMM time: " << elapsed << " sec\n";
    }
    #else

    std::cout << "Using reorder to do the conversion from float32 to bfloat16.\n";	

    // === oneDNN GEMM (bf16 with reorder) ===
    {
        std::vector<float> A_bf(M * K), B_bf(K * N), C_bf(M * N);
        A_bf = A;
        B_bf = B;

        engine eng(engine::kind::cpu, 0);
        stream s(eng);

        //auto start = std::chrono::high_resolution_clock::now();
        memory::dims a_dims = { M, K };
        memory::dims b_dims = { K, N };
        memory::dims c_dims = { M, N };

        auto a_md_f32   = memory::desc(a_dims, memory::data_type::f32, memory::format_tag::ab);
        auto b_md_f32   = memory::desc(b_dims, memory::data_type::f32, memory::format_tag::ab);
        auto c_md_f32   = memory::desc(c_dims, memory::data_type::f32, memory::format_tag::ab);

        auto a_md_bf16 = memory::desc(a_dims, memory::data_type::bf16, memory::format_tag::ab);
        auto b_md_bf16 = memory::desc(b_dims, memory::data_type::bf16, memory::format_tag::ab);
        auto c_md_bf16 = memory::desc(c_dims, memory::data_type::bf16, memory::format_tag::ab);

        auto A_f32_mem = memory(a_md_f32, eng, A_bf.data());
        auto B_f32_mem = memory(b_md_f32, eng, B_bf.data());
        auto C_f32_mem = memory(c_md_f32, eng, C_bf.data());

        auto start = std::chrono::high_resolution_clock::now();
        auto A_bf16_mem = memory(a_md_bf16, eng);
        auto B_bf16_mem = memory(b_md_bf16, eng);
        auto C_bf16_mem = memory(c_md_bf16, eng);       

        // === Reorder: BF16 -> F32 (for GEMM) ===
        //auto a_f32_md = memory::desc(a_dims, memory::data_type::f32, memory::format_tag::ab);
        //auto b_f32_md = memory::desc(b_dims, memory::data_type::f32, memory::format_tag::ab);
        //auto c_f32_md = memory::desc(c_dims, memory::data_type::f32, memory::format_tag::ab);

        //auto A_f32_mem = memory(a_f32_md, eng);
        //auto B_f32_mem = memory(b_f32_md, eng);
        //auto C_f32_mem = memory(c_f32_md, eng);

        // Reorder bf16 -> f32 (before GEMM)
        
        //auto start = std::chrono::high_resolution_clock::now();
	reorder(A_f32_mem,A_bf16_mem).execute(s, A_f32_mem, A_bf16_mem);
        reorder(B_f32_mem,B_bf16_mem).execute(s, B_f32_mem, B_bf16_mem);
        auto matmul_pd = matmul::primitive_desc(eng, a_md_bf16, b_md_bf16, c_md_bf16);
        auto matmul_prim = matmul(matmul_pd);

        //auto start = std::chrono::high_resolution_clock::now();
        //auto start = std::chrono::high_resolution_clock::now();  
        matmul_prim.execute(s, {
            {DNNL_ARG_SRC, A_bf16_mem},
            {DNNL_ARG_WEIGHTS, B_bf16_mem},
            {DNNL_ARG_DST, C_bf16_mem}
        });
        s.wait();
        auto end = std::chrono::high_resolution_clock::now();
                 
        //auto end = std::chrono::high_resolution_clock::now();
        //double elapsed = std::chrono::duration<double>(end - start).count();
        //std::cout << "oneDNN GEMM time (reorder + bf16): " << elapsed << " sec\n";

        // === Reorder back: F32 -> BF16 (after GEMM) ===
        reorder(C_bf16_mem, C_f32_mem).execute(s, C_bf16_mem, C_f32_mem);
        s.wait();

        //auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        std::cout << "oneDNN GEMM time (reorder + bf16): " << elapsed << " sec\n";

       
        // Optional: Convert result back to float32 for validation
        C_dnnl = C_bf;
    }	
    #endif 



    // === Compare Results ===
    double max_diff = 0.0;
    for (size_t i = 0; i < C_mkl.size(); ++i) {
        double diff = std::abs(C_mkl[i] - C_dnnl[i]);
        if (diff > max_diff) max_diff = diff;
    }
    std::cout << "Max diff (MKL vs oneDNN): " << max_diff << "\n";
}
