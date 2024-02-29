
#include <stdio.h>
#include <stdlib.h>
#include <supa.h>
#include <supa_runtime.h>
#include <vector>
#include <time.h>

#include "../include/sddmm_supa.hpp"
#include "../include/supa_util.hpp"


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
            flag = false;
            // return false;
        }
    }
    return flag;
}



bool sddmm_check(int size_n, int size_k) {
    size_t size_A = size_n * size_n * sizeof(float);
    size_t size_X = size_n * size_k * sizeof(float);
    size_t size_Y = size_n * size_k * sizeof(float);

    float *h_A = (float *)malloc(size_A);
    float *h_X = (float *)malloc(size_X);
    float *h_dY = (float *)malloc(size_Y);

    // float *h_dA_ref = (float *)malloc(size_X); //参考结果

    if (h_A == NULL || h_X == NULL || h_dY == NULL){
        printf ("malloc failed\n");
        return false;
    }
    init_random_sparse(h_A, size_n*size_n, 0.01);
    init_random(h_X, size_n*size_k);
    init_random(h_dY, size_n*size_k);

    std::vector<int> csrptr, csrind;
    std::vector<float> csrval;
    csrptr.push_back(0);
    for(int row=0;row < size_n;++row){
        for(int col=0;col < size_n;++col){
            int id = row*size_n+col;
            if(h_A[id] != 0){
                csrind.push_back(col);
                csrval.push_back(h_A[id]);
            }
        }
        csrptr.push_back(csrind.size());
    }
    int nnz = csrind.size();
    float* h_dA_ref = (float*)malloc(nnz * sizeof(float));

    size_t size_ptr = csrptr.size() * sizeof(int);
    size_t size_ind = csrind.size() * sizeof(int);
    size_t size_val = csrval.size() * sizeof(float);

    float *d_dY = NULL;
    checkSupaError(suMallocDevice((void **)&d_dY, size_Y));
    float *d_X = NULL;
    checkSupaError(suMallocDevice((void **)&d_X, size_X));
    // suMemset((void *)d_C, 0, size_C);
    int *d_ptr = NULL;
    checkSupaError(suMallocDevice((void **)&d_ptr, size_ptr));
    int *d_ind = NULL;
    checkSupaError(suMallocDevice((void **)&d_ind, size_ind));
    // DType *d_val = NULL;
    // checkSupaError(suMallocDevice((void **)&d_val, size_val));
    float* d_dA = NULL;
    checkSupaError(suMallocDevice((void **)&d_dA, nnz * sizeof(float)));
    suMemset((void*)d_dA, 0, nnz * sizeof(float));

    checkSupaError(suMemcpy(d_dY,h_dY,size_Y,suMemcpyHostToDevice));
    checkSupaError(suMemcpy(d_X,h_X,size_X,suMemcpyHostToDevice));
    checkSupaError(suMemcpy(d_ptr,&csrptr[0],size_ptr,suMemcpyHostToDevice));
    checkSupaError(suMemcpy(d_ind,&csrind[0],size_ind,suMemcpyHostToDevice));
    // checkSupaError(suMemcpy(d_val,&csrval[0],size_val,suMemcpyHostToDevice));

    if (size_k % 2 == 0) {
        suLaunchKernel(
            sddmmCSR2Scale, dim3(nnz / 16 + (nnz & 15), 1, 1), dim3(16, 4, 1), 0, NULL, 
            size_n, size_k, nnz, d_ptr, d_ind, d_dY, d_X, d_dA
        );
    } else {
        suLaunchKernel(
            sddmmCSR1Scale, dim3(nnz / 16 + (nnz & 15), 1, 1), dim3(32, 4, 1), 0, NULL,
            size_n, size_k, nnz, d_ptr, d_ind, d_dY, d_X, d_dA
        );
    }
}


int main() {
    srand((int)time(0));

    sddmm_check(1000, 128);
}


