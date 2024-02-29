#include <stdio.h>
#include <stdlib.h>
#include <supa.h>
#include <supa_runtime.h>
#include <vector>
#include <time.h>

#include "../include/spmm_supa.hpp"
#include "../include/supa_util.hpp"

enum method{
    naive,
    dgsparse_0,
    dgsparse_1,
    dgsparse_2,
    dgsparse_3
};

__global__ void warm_up() {}

// A * B =C
// A: sparse, m*k
// B: dense, k*n
// C: dense, m*n
void init_random_sparse(float *A, int size, float threshold){
    for(int i=0;i<size;i++){
        if(rand()/(float)RAND_MAX < threshold){
            // A[i] = rand()/(float)RAND_MAX;
            A[i] = 1;
        }
        else{
            A[i] = 0;
        }
    }
}

void init_random(float *A, int size){
    for(int i=0;i<size;i++){
        A[i] = rand()/(float)RAND_MAX;
    }
}

void transpose(float *from, float *to, int a, int b){
    for(int i = 0; i < a; ++i){
        for(int j = 0; j < b; ++j){
            to[j * a + i] = from[i * b + j];
        }
    }
}

bool check_result(float *C_ref, float *C, int num){
    bool flag = true;
    for (int i = 0; i < num; ++i)
    {
        if (C_ref[i] - C[i] > 1e-4)
        {
            printf("Result verification failed at element %d!\nC_ref[%d] = %f, C[%d] = %f\n", i, i, C_ref[i], i, C[i]);
            // flag = false;
            return false;
        }
    }
    return flag;
}

template <typename DType, typename DType2, typename DType4>
bool spmm_check(int size_m, int size_k, int size_n, method algorithm)
{
    size_t size_A = size_m * size_k * sizeof(DType);
    size_t size_B = size_n * size_k * sizeof(DType);
    size_t size_C = size_m * size_n * sizeof(DType);

    //申请Host内存并初始化
    DType *h_A = (DType *)malloc(size_A);
    DType *h_B = (DType *)malloc(size_B);
    DType *h_C = (DType *)malloc(size_C);

    DType *h_ref = (DType *)malloc(size_C); //参考结果

    if (h_A == NULL || h_B == NULL || h_C == NULL || h_ref == NULL){
        printf ("malloc failed\n");
        return false;
    }
    init_random_sparse(h_A, size_m*size_k, 0.01);
    init_random(h_B, size_k*size_n);

    std::vector<int> csrptr, csrind;
    std::vector<DType> csrval;
    csrptr.push_back(0);
    for(int row=0;row < size_m;++row){
        for(int col=0;col < size_k;++col){
            int id = row*size_k+col;
            if(h_A[id] != 0){
                csrind.push_back(col);
                csrval.push_back(h_A[id]);
            }
        }
        csrptr.push_back(csrind.size());
    }

    size_t size_ptr = csrptr.size() * sizeof(int);
    size_t size_ind = csrind.size() * sizeof(int);
    size_t size_val = csrval.size() * sizeof(DType);

    //申请Device内存
    DType *d_B = NULL;
    checkSupaError(suMallocDevice((void **)&d_B, size_B));
    DType *d_C = NULL;
    checkSupaError(suMallocDevice((void **)&d_C, size_C));
    suMemset((void *)d_C, 0, size_C);
    int *d_ptr = NULL;
    checkSupaError(suMallocDevice((void **)&d_ptr, size_ptr));
    int *d_ind = NULL;
    checkSupaError(suMallocDevice((void **)&d_ind, size_ind));
    DType *d_val = NULL;
    checkSupaError(suMallocDevice((void **)&d_val, size_val));

    //从Host端提交到Device端
    checkSupaError(suMemcpy(d_B,h_B,size_B,suMemcpyHostToDevice));
    checkSupaError(suMemcpy(d_ptr,&csrptr[0],size_ptr,suMemcpyHostToDevice));
    checkSupaError(suMemcpy(d_ind,&csrind[0],size_ind,suMemcpyHostToDevice));
    checkSupaError(suMemcpy(d_val,&csrval[0],size_val,suMemcpyHostToDevice));

    //调用kernel
    if (algorithm == method::naive){
        const int kblockSize = 256;
        int blockDimX = min(size_n, 32);
        int blockDimY = kblockSize / blockDimX;
        int gridSize = CEIL(size_m, blockDimY);
        int tile_k = CEIL(size_n, 32);
        suLaunchKernel(spmm_naive_kernel, 
                    dim3(gridSize, tile_k, 1), dim3(blockDimX, blockDimY, 1), 0, NULL,
                    size_m, size_n, d_ptr, d_ind, d_val, d_B, d_C
        );

    }else if(algorithm == method::dgsparse_0){
            int Mdim_worker = size_m;
            int Ndim_worker = size_n;

            int RefThreadPerBlock = 256;
            int Ndim_threadblock = CEIL(Ndim_worker, RefThreadPerBlock);
            int Ndim_thread_per_tb = min(Ndim_worker, RefThreadPerBlock);
            int Mdim_thread_per_tb = CEIL(RefThreadPerBlock, Ndim_thread_per_tb);
            int Mdim_threadblock = CEIL(Mdim_worker, Mdim_thread_per_tb);

            dim3 gridDim(Mdim_threadblock, Ndim_threadblock, 1);
            dim3 blockDim(Ndim_thread_per_tb, Mdim_thread_per_tb, 1);

            suLaunchKernel(csrspmm_seqreduce_rowbalance_kernel<int, DType>,
                            gridDim, blockDim, 0, NULL,
                            Mdim_worker, Ndim_worker, d_ptr, d_ind, d_val,
                            d_B, d_C);
    }
    else if(algorithm == method::dgsparse_1){
        int Mdim_worker = size_m;
        int Ndim_worker = size_n;

        int RefThreadPerBlock = 256;
        int Ndim_threadblock = CEIL(Ndim_worker, RefThreadPerBlock);
        int Ndim_thread_per_tb = min(Ndim_worker, RefThreadPerBlock);
        int Mdim_thread_per_tb = CEIL(RefThreadPerBlock, Ndim_thread_per_tb);
        int Mdim_threadblock = CEIL(Mdim_worker, Mdim_thread_per_tb);

        dim3 gridDim(Mdim_threadblock, Ndim_threadblock, 1);
        dim3 blockDim(Ndim_thread_per_tb, Mdim_thread_per_tb, 1);

        suLaunchKernel(csrspmm_seqreduce_nnzbalance_kernel<int, DType>,
                        gridDim, blockDim, 0, NULL,
                        Mdim_worker, Ndim_worker, csrval.size(), 
                        d_ptr, d_ind, d_val, d_B, d_C);
    }
    else if(algorithm == method::dgsparse_2){
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
            suLaunchKernel(csrspmm_parreduce_rowbalance_kernel<int, DType, DType4>,
                            gridDim, blockDim, 0, NULL,
                            Mdim_worker, Ndim_worker, 
                            d_ptr, d_ind, d_val, d_B, d_C);
        }
        if (coarsen_factor == 2){
            suLaunchKernel(csrspmm_parreduce_rowbalance_kernel<int, DType, DType2>,
                            gridDim, blockDim, 0, NULL,
                            Mdim_worker, Ndim_worker,
                            d_ptr, d_ind, d_val, d_B, d_C);
        }
        else {
            suLaunchKernel(csrspmm_parreduce_rowbalance_kernel<int, DType, DType>,
                            gridDim, blockDim, 0, NULL,
                            Mdim_worker, Ndim_worker, 
                            d_ptr, d_ind, d_val, d_B, d_C);
        }
    }
    else if(algorithm == method::dgsparse_3){
        int Ndim_worker = size_n;
        // factor of thread coarsening
        int coarsen_factor = (Ndim_worker % 4 == 0) ? 4 : (Ndim_worker % 2 == 0) ? 2 : 1;
        // number of parallel warps along M-dimension
        const int segreduce_size_per_warp = WARP_SIZE;
        int Nnzdim_worker = csrval.size(); // CEIL(spmatA.nnz, segreduce_size_per_warp);
        // partition large-N and map to blockdim.y to help cache performance
        int Ndim_threadblock = CEIL(Ndim_worker, WARP_SIZE);
        int Ndim_warp_per_tb = min(Ndim_worker, WARP_SIZE) / coarsen_factor;
        // int Ndim_warp_per_tb = min(N, WARP_SIZE)

        int RefThreadPerBlock = 256;
        int ref_warp_per_tb = RefThreadPerBlock / WARP_SIZE;
        int Nnzdim_warp_per_tb = CEIL(ref_warp_per_tb, Ndim_warp_per_tb);

        // total number of warps
        int gridDimX = CEIL(Nnzdim_worker, Nnzdim_warp_per_tb);
        int gridDimY = Ndim_threadblock;
        dim3 gridDim(gridDimX, gridDimY, 1);
        dim3 blockDim(Ndim_warp_per_tb * WARP_SIZE, Nnzdim_warp_per_tb, 1);

        if (coarsen_factor == 4) {
            suLaunchKernel(csrspmm_parreduce_nnzbalance_kernel<int,DType,DType4>,
                            gridDim, blockDim, 0, NULL,
                            size_m, Ndim_worker, Nnzdim_worker, 
                            d_ptr, d_ind, d_val, d_B, d_C);
        } else if (coarsen_factor == 2) {
            suLaunchKernel(csrspmm_parreduce_nnzbalance_kernel<int,DType,DType2>,
                            gridDim, blockDim, 0, NULL,
                            size_m, Ndim_worker, Nnzdim_worker, 
                            d_ptr, d_ind, d_val, d_B, d_C);
        } else {
            suLaunchKernel(csrspmm_parreduce_nnzbalance_kernel<int,DType,DType>,
                            gridDim, blockDim, 0, NULL,
                            size_m, Ndim_worker, Nnzdim_worker, 
                            d_ptr, d_ind, d_val, d_B, d_C);
        }
    }

    //CPU计算
    for(int row = 0; row<size_m;row++){
        for(int col = 0; col<size_n;col++){
            int id = row*size_n+col;
            h_ref[id]=0;
            for(int k = 0; k<size_k;k++){
                h_ref[id] += h_A[row * size_k + k] * h_B[k * size_n + col];
            }
        }
    }

    //将结果从Device端传回Host端
    checkSupaError(suMemcpy(h_C,d_C,size_C,suMemcpyDeviceToHost));
    checkSupaError(suDeviceSynchronize());

    bool flag = false;
    if(check_result(h_ref, h_C, size_m * size_n)){
        printf("\n###########spmm check pass!#############\n");
        flag = true;
    }


    // 释放内存
    suFree(d_B);
    suFree(d_C);
    suFree(d_ptr);
    suFree(d_ind);
    suFree(d_val);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_ref);

    csrptr.clear();csrptr.shrink_to_fit();
    csrind.clear();csrind.shrink_to_fit();
    csrval.clear();csrval.shrink_to_fit();
    return flag;
}

int main(int argc,char **argv)
{
    srand((int)time(0));

    //check correctness
    spmm_check<float, float2, float4>(1000, 1000, 128, method::naive);
    spmm_check<float, float2, float4>(1000, 1000, 128, method::dgsparse_0);
    spmm_check<float, float2, float4>(1000, 1000, 128, method::dgsparse_1);
    spmm_check<float, float2, float4>(1000, 1000, 128, method::dgsparse_2);
    spmm_check<float, float2, float4>(1000, 1000, 128, method::dgsparse_3);

    return 0;
}