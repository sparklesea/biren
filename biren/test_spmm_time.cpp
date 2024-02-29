#include <stdio.h>
#include <stdlib.h>
#include <supa.h>
#include <supa_runtime.h>
#include <vector>
#include <time.h>
#include <fstream>
#include <string>

#include "../include/spmm_supa.hpp"
#include "../include/dataloader.hpp"
#include "../include/ramArray.hpp"
#include "../include/mmio.hpp"

__global__ void warm_up() {}

// A * B =C
// A: sparse, m*k
// B: dense, k*n
// C: dense, m*n

template<typename DType, typename DType2, typename DType4>
float spmm_time(int size_m, int size_k, int size_n, int iter, 
                int algorithm, float *d_val, int *d_ptr, 
                int *d_ind, int nnz, float *d_B, float *d_C)
{
    suEvent_t start, end;
    float time_elapsed;
    suEventCreate(&start);
    suEventCreate(&end);

    for (int i = 0; i < 1000; i++)
        suLaunchKernel(warm_up, 1, 1, 0, NULL);

    //调用kernel
    if(algorithm == -2){
        const int kblockSize = 256;
        int blockDimX = min(size_n, 32);
        int blockDimY = kblockSize / blockDimX;
        int gridSize = CEIL(size_m, blockDimY);
        int tile_k = CEIL(size_n, 32);
        suEventRecord(start);
        for(int circle = 0; circle < iter; ++circle){
            suLaunchKernel(spmm_shmem_kernel,
                        dim3(gridSize, tile_k), dim3(32, blockDimY),
                        32*blockDimY*(sizeof(int)+sizeof(float)), NULL,
                        size_m, size_n, d_ptr, d_ind, d_val, d_B, d_C);
        }
        suEventRecord(end);
        suEventSynchronize(end);
        suEventElapsedTime(&time_elapsed, start, end);
    }
    if(algorithm == -1){
        const int kblockSize = 256;
        int blockDimX = min(size_n, 32);
        int blockDimY = kblockSize / blockDimX;
        int gridSize = CEIL(size_m, blockDimY);
        int tile_k = CEIL(size_n, 32);
        suEventRecord(start);
        for(int circle = 0; circle < iter; ++circle){
            suLaunchKernel(spmm_naive_kernel, 
                        dim3(gridSize, tile_k, 1), dim3(blockDimX, blockDimY, 1), 0, NULL,
                        size_m, size_n, d_ptr, d_ind, d_val, d_B, d_C
            );
        }
        suEventRecord(end);
        suEventSynchronize(end);
        suEventElapsedTime(&time_elapsed, start, end);
    }
    else if(algorithm == 0){
        int Mdim_worker = size_m;
        int Ndim_worker = size_n;

        int RefThreadPerBlock = 256;
        int Ndim_threadblock = CEIL(Ndim_worker, RefThreadPerBlock);
        int Ndim_thread_per_tb = min(Ndim_worker, RefThreadPerBlock);
        int Mdim_thread_per_tb = CEIL(RefThreadPerBlock, Ndim_thread_per_tb);
        int Mdim_threadblock = CEIL(Mdim_worker, Mdim_thread_per_tb);

        dim3 gridDim(Mdim_threadblock, Ndim_threadblock, 1);
        dim3 blockDim(Ndim_thread_per_tb, Mdim_thread_per_tb, 1);

        suEventRecord(start);
        for(int circle = 0; circle < iter; ++circle){
            suLaunchKernel(csrspmm_seqreduce_rowbalance_kernel<int, DType>,
                            gridDim, blockDim, 0, NULL,
                            Mdim_worker, Ndim_worker, d_ptr, d_ind, d_val,
                            d_B, d_C);
        }
        suEventRecord(end);
        suEventSynchronize(end);
        suEventElapsedTime(&time_elapsed, start, end);

        // printf("dgsparse_0 = %.3fms\n", time_elapsed/iter);
    }
    // else if(algorithm == 1){
    //     int Mdim_worker = size_m;
    //     int Ndim_worker = size_n;

    //     int RefThreadPerBlock = 256;
    //     int Ndim_threadblock = CEIL(Ndim_worker, RefThreadPerBlock);
    //     int Ndim_thread_per_tb = min(Ndim_worker, RefThreadPerBlock);
    //     int Mdim_thread_per_tb = CEIL(RefThreadPerBlock, Ndim_thread_per_tb);
    //     int Mdim_threadblock = CEIL(Mdim_worker, Mdim_thread_per_tb);

    //     dim3 gridDim(Mdim_threadblock, Ndim_threadblock, 1);
    //     dim3 blockDim(Ndim_thread_per_tb, Mdim_thread_per_tb, 1);

    //     suEventRecord(start);
    //     for(int circle = 0; circle < iter; ++circle){
    //         suLaunchKernel(csrspmm_seqreduce_nnzbalance_kernel<int, DType>,
    //                         gridDim, blockDim, 0, NULL,
    //                         Mdim_worker, Ndim_worker, nnz, 
    //                         d_ptr, d_ind, d_val, d_B, d_C);
    //     }
    //     suEventRecord(end);
    //     suEventSynchronize(end);
    //     suEventElapsedTime(&time_elapsed, start, end);

    //     // printf("dgsparse_1 = %.3fms\n", time_elapsed/iter);
    // }
    else if(algorithm == 1){
        int Mdim_worker = size_m;
        int Ndim_worker = size_n;
        int coarsen_factor = (Ndim_worker % 4 == 0) ? 4 : (Ndim_worker % 2 == 0) ? 2 : 1;
        // partition large-N and map to blockdim.y to help cache performance
        int RefThreadPerBlock = 256;

        int Ndim_threadblock = CEIL(Ndim_worker, WARP_SIZE);
        int Ndim_warp_per_tb = min(Ndim_worker, WARP_SIZE) / coarsen_factor;
        int ref_warp_per_tb = RefThreadPerBlock / WARP_SIZE;
        int Mdim_warp_per_tb = CEIL(ref_warp_per_tb, Ndim_warp_per_tb);

        int gridDimX = CEIL(Mdim_worker, Mdim_warp_per_tb);
        int gridDimY = Ndim_threadblock;

        dim3 gridDim(gridDimX, gridDimY, 1);
        dim3 blockDim(Ndim_warp_per_tb * WARP_SIZE, Mdim_warp_per_tb, 1);

        if (coarsen_factor == 4){
            suEventRecord(start);
            for(int circle = 0; circle < iter; ++circle){
                suLaunchKernel(csrspmm_parreduce_rowbalance_kernel<int, DType, DType4>,
                                gridDim, blockDim, 0, NULL,
                                Mdim_worker, Ndim_worker, 
                                d_ptr, d_ind, d_val, d_B, d_C);
            }
            suEventRecord(end);
        }
        else if (coarsen_factor == 2){
            suEventRecord(start);
            for(int circle = 0; circle < iter; ++circle){
                suLaunchKernel(csrspmm_parreduce_rowbalance_kernel<int, DType, DType2>,
                                gridDim, blockDim, 0, NULL,
                                Mdim_worker, Ndim_worker,
                                d_ptr, d_ind, d_val, d_B, d_C);
            }
            suEventRecord(end);
        }
        else {
            suEventRecord(start);
            for(int circle = 0; circle < iter; ++circle){
                suLaunchKernel(csrspmm_parreduce_rowbalance_kernel<int, DType, DType>,
                                gridDim, blockDim, 0, NULL,
                                Mdim_worker, Ndim_worker, 
                                d_ptr, d_ind, d_val, d_B, d_C);
            }
            suEventRecord(end);
        }
        suEventSynchronize(end);
        suEventElapsedTime(&time_elapsed, start, end);
    }

    checkSupaError(suDeviceSynchronize());

    return time_elapsed/iter;
}

void spmm_reference_host(int M,       // number of A-rows
                         int feature, // number of B_columns
                         int *csr_indptr, int *csr_indices,
                         float *csr_values, // three arrays of A's CSR format
                         float *B,          // assume row-major
                         float *C_ref)      // assume row-major
{
  for (int64_t i = 0; i < M; i++) {
    int begin = csr_indptr[i];
    int end = csr_indptr[i + 1];
    for (int p = begin; p < end; p++) {
      int k = csr_indices[p];
      float val = csr_values[p];
      for (int64_t j = 0; j < feature; j++) {
        C_ref[i * feature + j] += val * B[k * feature + j];
      }
    }
  }
}

bool check_result(int M, int N, float *C, float *C_ref) {
  bool passed = true;
  for (int64_t i = 0; i < M; i++) {
    for (int64_t j = 0; j < N; j++) {
      float c = C[i * N + j];
      float c_ref = C_ref[i * N + j];
      if (fabs(c - c_ref) > 1e-2 * fabs(c_ref)) {
        printf(
            "Wrong result: i = %ld, j = %ld, result = %lf, reference = %lf.\n",
            i, j, c, c_ref);

        passed = false;
        // break;
        return false;
      }
    }
  }
  return passed;
}

int main(int argc,char **argv)
{
    if(argc<3){
        printf("Input: first get the path of sparse matrix, then get the "
            "feature length of dense matrix\n");
        exit(1);
    }
    char *filepath = argv[1];
    int feature_size = atoi(argv[2]);
    bool DEBUG = false;
    const int ITER = 20;
    srand((int)time(0));
    printf("%s,", argv[1]);

    auto sp_mat = DataLoader<int, float>(filepath);

    int M=sp_mat.nrow;
    int N=feature_size;
    int K=sp_mat.ncol;
    // printf("%d, %d, %d\n", M, N, K);
    util::RamArray<float> dn_mat(K * N);
    util::RamArray<float> out_mat(M * N);

    // dn_mat.fill_random_h();
    dn_mat.fill_default_one();
    out_mat.fill_zero_h();

    dn_mat.upload();
    out_mat.upload();
    sp_mat.upload();

    // warm up
    for(int method=-2;method<2;++method){

        //check correctness
        float test_time = spmm_time<float, float2, float4>(M, K, N, ITER, method,
                        sp_mat.sp_data.d_array.get(), 
                        sp_mat.sp_csrptr.d_array.get(), 
                        sp_mat.sp_csrind.d_array.get(), 
                        sp_mat.nnz, dn_mat.d_array.get(),
                        out_mat.d_array.get());
        if(DEBUG){
            util::RamArray<float> out_ref(M * N);
            out_ref.fill_zero_h();
            spmm_reference_host(M, N, sp_mat.sp_csrptr.h_array.get(),
                                sp_mat.sp_csrind.h_array.get(),
                                sp_mat.sp_data.h_array.get(),
                                dn_mat.h_array.get(),
                                out_ref.h_array.get());
            out_mat.download();
            if(check_result(M, N, out_mat.h_array.get(), out_ref.h_array.get()))
                printf("check passed!\n");
        }
        double gflop=(double)sp_mat.nnz*2/1000000*N;

        printf("%f,%f,", test_time, gflop/(test_time));
    }
    printf("\n");
    return 0;
}
