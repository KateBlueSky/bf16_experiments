#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <dnnl.hpp>
#include <mkl.h>

using namespace dnnl;

// Fixed M and N
constexpr int M = 1024;
constexpr int N = 100;
constexpr int K = 20;


int main() {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    //for (int K = 20; K <= 100; K += 10) {
        std::cout << "\nBenchmarking GEMM: " << M << "x" << K << " x " << K << "x" << N << "\n";

        std::vector<float> A(M * K), B(K * N), C_mkl(M * N), C_dnnl(M * N);

        for (auto& x : A) x = dist(gen);
        for (auto& x : B) x = dist(gen);

        // === MKL GEMM (float32) ===
        {
            std::fill(C_mkl.begin(), C_mkl.end(), 0.0f);
            auto start = std::chrono::high_resolution_clock::now();

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        M, N, K,
                        1.0f,
                        A.data(), K,
                        B.data(), N,
                        0.0f,
                        C_mkl.data(), N);

            auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(end - start).count();
            std::cout << "MKL time: " << elapsed << " sec\n";
        }

        // === oneDNN GEMM ===
        {
            std::vector<float> A_bf = A, B_bf = B, C_bf(M * N);

            engine eng(engine::kind::cpu, 0);
            stream s(eng);

            memory::dims a_dims = { M, K };
            memory::dims b_dims = { K, N };
            memory::dims c_dims = { M, N };

            auto a_md = memory::desc(a_dims, memory::data_type::f32, memory::format_tag::ab);
            auto b_md = memory::desc(b_dims, memory::data_type::f32, memory::format_tag::ab);
            auto c_md = memory::desc(c_dims, memory::data_type::f32, memory::format_tag::ab);

            auto A_mem = memory(a_md, eng, A_bf.data());
            auto B_mem = memory(b_md, eng, B_bf.data());
            auto C_mem = memory(c_md, eng, C_bf.data());

            auto start = std::chrono::high_resolution_clock::now();
            auto matmul_pd = matmul::primitive_desc(eng, a_md, b_md, c_md);
            auto matmul_prim = matmul(matmul_pd);

            matmul_prim.execute(s, {
                {DNNL_ARG_SRC, A_mem},
                {DNNL_ARG_WEIGHTS, B_mem},
                {DNNL_ARG_DST, C_mem}
            });
            s.wait();
            auto end = std::chrono::high_resolution_clock::now();

            double elapsed = std::chrono::duration<double>(end - start).count();
            std::cout << "oneDNN time: " << elapsed << " sec\n";

            C_dnnl = C_bf;
        }

        // === Compare Results ===
        float rms_diff = 0.0f;
        float max_diff = 0.0f;
        for (size_t i = 0; i < C_mkl.size(); ++i) {
            float diff = C_mkl[i] - C_dnnl[i];
            rms_diff += diff * diff;
            max_diff = std::max(max_diff, std::abs(diff));
        }
        rms_diff = std::sqrt(rms_diff / C_mkl.size());

        std::cout << "RMS Diff: " << rms_diff << "\n";
        std::cout << "Max Diff: " << max_diff << "\n";
    //}

    return 0;
}
