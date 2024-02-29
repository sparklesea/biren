
#include <stdio.h>
#include <stdlib.h>
#include <supa.h>
#include <supa_runtime.h>
#include <vector>
#include <time.h>
#include <fstream>
#include <string>

#include "../include/sddmm_supa.hpp"
#include "../include/dataloader.hpp"
#include "../include/ramArray.hpp"
#include "../include/mmio.hpp"

__global__ void warm_up() {}

float sddmm_time(
    int size_m, int size_k, int size_n, int iter, int algorithm, 
    int* d_ptr, int* d_ind, int nnz, float* d_B, float* d_C, float* d_out
) {
    suEvent_t start, end;
    float time_elapsed;
    suEventCreate(&start);
    suEventCreate(&end);

    for (int i = 0; i < 1000; i++)
        suLaunchKernel(warm_up, 1, 1, 0, NULL);

    if (algorithm == -1) {
       suEventRecord(start);
       for (int circle = 0; circle < iter; circle++) {
           suLaunchKernel(
               sddmmCSRNaive, dim3((nnz + 256 - 1) / 256, 1, 1), dim3(256, 1, 1), 0, NULL, 
               size_m, size_k, nnz, d_ptr, d_ind, d_B, d_C, d_out
           );
       }
       suEventRecord(end);
       suEventSynchronize(end);
       suEventElapsedTime(&time_elapsed, start, end);
    } else if (algorithm == 0) {
        if (size_k % 2 == 0) {
            suEventRecord(start);
            for (int circle = 0; circle < iter; circle++) {
                suLaunchKernel(
                    sddmmCSR2Scale, dim3(nnz / 16 + (nnz & 15), 1, 1), dim3(16, 4, 1), 0, NULL, 
                    size_m, size_k, nnz, d_ptr, d_ind, d_B, d_C, d_out
                );
            }
            suEventRecord(end);
            suEventSynchronize(end);
            suEventElapsedTime(&time_elapsed, start, end);
        } else {
            suEventRecord(start);
            for (int circle = 0; circle < iter; circle++) {
                suLaunchKernel(
                    sddmmCSR1Scale, dim3(nnz / 16 + (nnz & 15), 1, 1), dim3(32, 4, 1), 0, NULL, 
                    size_m, size_k, nnz, d_ptr, d_ind, d_B, d_C, d_out
                );
            }
            suEventRecord(end);
            suEventSynchronize(end);
            suEventElapsedTime(&time_elapsed, start, end);
        }
    }
    checkSupaError(suDeviceSynchronize());
    return time_elapsed/iter;
}

void sddmm_reference_host(
    int M, int N, int K, 
    int* rowPtr, int* colIdx, 
    float* B, float* C, float* out
) {
    int index = 0;
    int row = 0;
    while (row < M) {
        int num = rowPtr[row + 1] - rowPtr[row];
        while (num) {
            int col = colIdx[index];
            for (int i = 0; i < K; i++) {
                out[index] += B[row * K + i] * C[col * K + i];
            }
            index++;
            num--;
        }
        row++;
    }
}

bool check_result(
    float* C_ref, float* C, int nnz
) {
    for (int i = 0; i < nnz; i++) {
        if (fabs(C_ref[i] - C[i]) > 1e-2 * fabs(C_ref[i])) {
            printf("Wrong result: ref = %f, result = %f \n", C_ref[i], C[i]);
            return false;
        }
    }
    return true;
}

int main(int argc,char **argv) {
    if(argc<3){
        printf("Input: first get the path of sparse matrix, then get the "
            "feature length of dense matrix\n");
        exit(1);
    }
    char *filepath = argv[1];
    printf("%s,", filepath);
    int K = atoi(argv[2]);
    bool DEBUG = false;
    const int ITER = 10;
    srand((int)time(0));

    auto sp_mat = DataLoader<int, float>(filepath);
    int M = sp_mat.nrow;
    int N = sp_mat.ncol;
    util::RamArray<float> B_mat(M * K);
    util::RamArray<float> C_mat(N * K);
    util::RamArray<float> out_mat(sp_mat.nnz);

    B_mat.fill_random_h();
    C_mat.fill_random_h();
    out_mat.fill_zero_h();

    sp_mat.upload();
    B_mat.upload();
    C_mat.upload();
    out_mat.upload();

    std::vector<float> speed;

    for (int method = -1; method <= 0; method++) {
        float test_time = sddmm_time(
            M, K, N, ITER, method, 
            sp_mat.sp_csrptr.d_array.get(), 
            sp_mat.sp_csrind.d_array.get(), 
            sp_mat.nnz, 
            B_mat.d_array.get(), C_mat.d_array.get(), out_mat.d_array.get()
        );
        speed.push_back(test_time);
        if (DEBUG) {
            util::RamArray<float> out_ref(sp_mat.nnz);
            out_ref.fill_zero_h();
            sddmm_reference_host(
                M, N, K, 
                sp_mat.sp_csrptr.h_array.get(), 
                sp_mat.sp_csrind.h_array.get(), 
                B_mat.h_array.get(), C_mat.h_array.get(), out_ref.h_array.get()
            );
            out_mat.download();
            if(check_result(out_ref.h_array.get(), out_mat.h_array.get(), sp_mat.nnz))
                printf("K = %d, method = %d check passed!\n", K, method);
        }
    }
    printf("%f,%f,", speed[0], speed[1]);
    printf("\n");
}

