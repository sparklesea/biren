#include <iostream>
#include <tuple>
#include <vector>

#include "../../include/cuda/csr2csc.cuh"
#include "../../include/cuda/cuda_util.cuh"
#include "../../include/cuda/sddmm_cuda.cuh"
#include "../../include/cuda/spmm_cuda.cuh"
#include "../../include/gspmm.h"

std::vector<torch::Tensor>
spmm_cuda(torch::Tensor csrptr, torch::Tensor indices, torch::Tensor edge_val,
          torch::Tensor in_feat, bool has_value, int64_t algorithm,
          REDUCEOP reduce_op, COMPUTEOP compute_op) {
  //   assertTensor(csrptr, torch::kInt32);
  //   assertTensor(indices, torch::kInt32);
  //   assertTensor(in_feat, torch::kFloat32);
  //   assertTensor(edge_val, torch::kFloat32);
  in_feat = in_feat.contiguous();
  int v = csrptr.size(0) - 1;
  int Ndim_worker = in_feat.size(1);
  int f = Ndim_worker;
  int e = indices.size(0);
  auto devid = in_feat.device().index();

  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto out_feat = torch::empty({v, f}, options);
  auto options_E =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, devid);
  auto out_E = torch::empty({v, f}, options_E);

  if (algorithm == 0) {
    int Mdim_worker = csrptr.size(0) - 1;
    // int v = Mdim_worker;
    int Ndim_worker = in_feat.size(1);
    // int f = Ndim_worker;
    // int e = indices.size(0);
    int RefThreadPerBlock = 256;
    int Ndim_threadblock = CEIL(Ndim_worker, RefThreadPerBlock);
    int Ndim_thread_per_tb = min(Ndim_worker, RefThreadPerBlock);
    int Mdim_thread_per_tb = CEIL(RefThreadPerBlock, Ndim_thread_per_tb);
    int Mdim_threadblock = CEIL(Mdim_worker, Mdim_thread_per_tb);

    dim3 gridDim(Mdim_threadblock, Ndim_threadblock, 1);
    dim3 blockDim(Ndim_thread_per_tb, Mdim_thread_per_tb, 1);

    // auto out_feat = torch::empty({v, f}, options);

    if (has_value)
      SWITCH_REDUCEOP(reduce_op, REDUCE, {
        SWITCH_COMPUTEOP(compute_op, COMPUTE, {
          csrspmm_seqreduce_rowbalance_kernel<int, float, REDUCE, COMPUTE>
              <<<gridDim, blockDim>>>(
                  Mdim_worker, Ndim_worker, csrptr.data_ptr<int>(),
                  indices.data_ptr<int>(), edge_val.data_ptr<float>(),
                  in_feat.data_ptr<float>(), out_feat.data_ptr<float>(),
                  out_E.data_ptr<int>());
        });
      });

    else
      SWITCH_REDUCEOP(reduce_op, REDUCE, {
        SWITCH_COMPUTEOP(compute_op, COMPUTE, {
          csrspmm_seqreduce_rowbalance_kernel<int, float, REDUCE, COMPUTE>
              <<<gridDim, blockDim>>>(
                  Mdim_worker, Ndim_worker, csrptr.data_ptr<int>(),
                  indices.data_ptr<int>(), (float *)nullptr,
                  in_feat.data_ptr<float>(), out_feat.data_ptr<float>(),
                  out_E.data_ptr<int>());
        });
      });
  } else if (algorithm == 1) {
    // int Mdim_worker = csrptr.size(0) - 1;
    // int v = Mdim_worker;
    int Nnzdim_worker = indices.size(0);
    // int Nnzdim_worker = csrptr.size(0) * 32;
    // int v = csrptr.size(0);
    int Ndim_worker = in_feat.size(1);
    // int f = Ndim_worker;
    // int e = indices.size(0);
    int RefThreadPerBlock = 256;
    int Ndim_threadblock = CEIL(Ndim_worker, RefThreadPerBlock);
    int Ndim_thread_per_tb = min(Ndim_worker, RefThreadPerBlock);
    int Nnzdim_thread_per_tb = CEIL(RefThreadPerBlock, Ndim_thread_per_tb);
    int Nnzdim_threadblock = CEIL(Nnzdim_worker, Nnzdim_thread_per_tb);

    dim3 gridDim(Nnzdim_threadblock, Ndim_threadblock, 1);
    dim3 blockDim(Ndim_thread_per_tb, Nnzdim_thread_per_tb, 1);

    if (has_value)
      SWITCH_REDUCEOP(reduce_op, REDUCE, {
        SWITCH_COMPUTEOP(compute_op, COMPUTE, {
          csrspmm_seqreduce_nnzbalance_kernel<int, float, REDUCE, COMPUTE>
              <<<gridDim, blockDim>>>(
                  v, Ndim_worker, Nnzdim_worker, csrptr.data_ptr<int>(),
                  indices.data_ptr<int>(), edge_val.data_ptr<float>(),
                  in_feat.data_ptr<float>(), out_feat.data_ptr<float>(),
                  out_E.data_ptr<int>());
        });
      });
    else
      SWITCH_REDUCEOP(reduce_op, REDUCE, {
        SWITCH_COMPUTEOP(compute_op, COMPUTE, {
          csrspmm_seqreduce_nnzbalance_kernel<int, float, REDUCE, COMPUTE>
              <<<gridDim, blockDim>>>(
                  v, Ndim_worker, Nnzdim_worker, csrptr.data_ptr<int>(),
                  indices.data_ptr<int>(), (float *)nullptr,
                  in_feat.data_ptr<float>(), out_feat.data_ptr<float>(),
                  out_E.data_ptr<int>());
        });
      });
  } else if (algorithm == 2) {
    // number of parallel warps along M-dimension
    int Mdim = csrptr.size(0) - 1;
    int Ndim = in_feat.size(1);
    int coarsen_factor = (Ndim % 4 == 0) ? 4 : (Ndim % 2 == 0) ? 2 : 1;
    // partition large-N and map to blockdim.y to help cache performance
    int Ndim_threadblock = CEIL(in_feat.size(1), 32);
    int Ndim_warp_per_tb = min(Ndim, 32) / coarsen_factor;
    int RefThreadPerBlock = 256;
    int ref_warp_per_tb = RefThreadPerBlock / 32;
    int Mdim_warp_per_tb = CEIL(ref_warp_per_tb, Ndim_warp_per_tb);

    // total number of warps
    int gridDimX = CEIL(Mdim, Mdim_warp_per_tb);
    int gridDimY = Ndim_threadblock;

    dim3 gridDim(gridDimX, gridDimY, 1);
    dim3 blockDim(Ndim_warp_per_tb * 32, Mdim_warp_per_tb, 1);

    if (has_value)
      if (coarsen_factor == 4) {
        SWITCH_REDUCEOP(reduce_op, REDUCE, {
          SWITCH_COMPUTEOP(compute_op, COMPUTE, {
            csrspmm_parreduce_rowbalance_kernel<int, float, float4, REDUCE,
                                                COMPUTE><<<gridDim, blockDim>>>(
                Mdim, Ndim, csrptr.data_ptr<int>(), indices.data_ptr<int>(),
                edge_val.data_ptr<float>(), in_feat.data_ptr<float>(),
                out_feat.data_ptr<float>(), out_E.data_ptr<int>());
          });
        });
      } else if (coarsen_factor == 2) {
        SWITCH_REDUCEOP(reduce_op, REDUCE, {
          SWITCH_COMPUTEOP(compute_op, COMPUTE, {
            csrspmm_parreduce_rowbalance_kernel<int, float, float2, REDUCE,
                                                COMPUTE><<<gridDim, blockDim>>>(
                Mdim, Ndim, csrptr.data_ptr<int>(), indices.data_ptr<int>(),
                edge_val.data_ptr<float>(), in_feat.data_ptr<float>(),
                out_feat.data_ptr<float>(), out_E.data_ptr<int>());
          });
        });
      } else {
        SWITCH_REDUCEOP(reduce_op, REDUCE, {
          SWITCH_COMPUTEOP(compute_op, COMPUTE, {
            csrspmm_parreduce_rowbalance_kernel<int, float, float, REDUCE,
                                                COMPUTE><<<gridDim, blockDim>>>(
                Mdim, Ndim, csrptr.data_ptr<int>(), indices.data_ptr<int>(),
                edge_val.data_ptr<float>(), in_feat.data_ptr<float>(),
                out_feat.data_ptr<float>(), out_E.data_ptr<int>());
          });
        });
      }
    else {
      if (coarsen_factor == 4) {
        SWITCH_REDUCEOP(reduce_op, REDUCE, {
          SWITCH_COMPUTEOP(compute_op, COMPUTE, {
            csrspmm_parreduce_rowbalance_kernel<int, float, float4, REDUCE,
                                                COMPUTE><<<gridDim, blockDim>>>(
                Mdim, Ndim, csrptr.data_ptr<int>(), indices.data_ptr<int>(),
                (float *)nullptr, in_feat.data_ptr<float>(),
                out_feat.data_ptr<float>(), out_E.data_ptr<int>());
          });
        });
      } else if (coarsen_factor == 2) {
        SWITCH_REDUCEOP(reduce_op, REDUCE, {
          SWITCH_COMPUTEOP(compute_op, COMPUTE, {
            csrspmm_parreduce_rowbalance_kernel<int, float, float2, REDUCE,
                                                COMPUTE><<<gridDim, blockDim>>>(
                Mdim, Ndim, csrptr.data_ptr<int>(), indices.data_ptr<int>(),
                (float *)nullptr, in_feat.data_ptr<float>(),
                out_feat.data_ptr<float>(), out_E.data_ptr<int>());
          });
        });
      } else {
        SWITCH_REDUCEOP(reduce_op, REDUCE, {
          SWITCH_COMPUTEOP(compute_op, COMPUTE, {
            csrspmm_parreduce_rowbalance_kernel<int, float, float, REDUCE,
                                                COMPUTE><<<gridDim, blockDim>>>(
                Mdim, Ndim, csrptr.data_ptr<int>(), indices.data_ptr<int>(),
                (float *)nullptr, in_feat.data_ptr<float>(),
                out_feat.data_ptr<float>(), out_E.data_ptr<int>());
          });
        });
      }
    }
  }
  else if(algorithm==3)
  {
    int N=in_feat.size(1);
    // factor of thread coarsening
    int coarsen_factor = (N % 4 == 0) ? 4 : (N % 2 == 0) ? 2 : 1;
    // number of parallel warps along M-dimension
    const int segreduce_size_per_warp = 32;
    int Nnzdim_worker = indices.size(0); // CEIL(spmatA.nnz,
    segreduce_size_per_warp);
    // partition large-N and map to blockdim.y to help cache performance
    int Ndim_threadblock = CEIL(N, 32);
    int Ndim_warp_per_tb = min(N, 32) / coarsen_factor;
    // int Ndim_warp_per_tb = min(N, 32)

    int RefThreadPerBlock = 256;
    int ref_warp_per_tb = RefThreadPerBlock / 32;
    int Nnzdim_warp_per_tb = CEIL(ref_warp_per_tb, Ndim_warp_per_tb);

    // total number of warps
    int gridDimX = CEIL(Nnzdim_worker, Nnzdim_warp_per_tb);
    int gridDimY = Ndim_threadblock;
    dim3 gridDim(gridDimX, gridDimY, 1);
    dim3 blockDim(Ndim_warp_per_tb * 32, Nnzdim_warp_per_tb, 1);

    if (coarsen_factor == 4) {
      csrspmm_parreduce_nnzbalance_kernel<int,float,float4><<<gridDim,
      blockDim>>>(
          v, N, Nnzdim_worker, csrptr.data_ptr<int>(),
          indices.data_ptr<int>(), edge_val.data_ptr<float>(),
          in_feat.data_ptr<float>(), out_feat.data_ptr<float>());
    } else if (coarsen_factor == 2) {
      csrspmm_parreduce_nnzbalance_kernel<int,float,float2><<<gridDim,
      blockDim>>>(
          v, N, Nnzdim_worker, csrptr.data_ptr<int>(),
          indices.data_ptr<int>(), edge_val.data_ptr<float>(),
          in_feat.data_ptr<float>(), out_feat.data_ptr<float>());
    } else {
      csrspmm_parreduce_nnzbalance_kernel<int,float,float><<<gridDim,
      blockDim>>>(
          v, N, Nnzdim_worker, csrptr.data_ptr<int>(),
          indices.data_ptr<int>(), edge_val.data_ptr<float>(),
          in_feat.data_ptr<float>(), out_feat.data_ptr<float>());
    }
  }

  SWITCH_REDUCEOP(reduce_op, REDUCE, {
    if (REDUCE::Op == MAX || REDUCE::Op == MIN) {
      return {out_feat, out_E};
    } else {
      return {out_feat};
    }
  });
}

// torch::Tensor sddmm_cuda_coo(torch::Tensor rowind, torch::Tensor colind,
//                              torch::Tensor D1, torch::Tensor D2) {
//   D1 = D1.contiguous();
//   D2 = D2.contiguous();
//   const auto k = D1.size(1);
//   const auto nnz = rowind.size(0);
//   auto devid = D1.device().index();
//   auto options =
//       torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
//   auto out = torch::zeros({nnz}, options);
//   if ((k % 4) == 0) {
//     sddmmCOO4Scale<<<dim3(nnz / 16 + (nnz & 15), 1, 1), dim3(8, 4, 1)>>>(
//         k, nnz, rowind.data_ptr<int>(), colind.data_ptr<int>(),
//         D1.data_ptr<float>(), D2.data_ptr<float>(), out.data_ptr<float>());
//   } else if ((k % 2) == 0) {
//     sddmmCOO2Scale<<<dim3(nnz / 16 + (nnz & 15), 1, 1), dim3(16, 4, 1)>>>(
//         k, nnz, rowind.data_ptr<int>(), colind.data_ptr<int>(),
//         D1.data_ptr<float>(), D2.data_ptr<float>(), out.data_ptr<float>());
//   } else {
//     sddmmCOO1Scale<<<dim3(nnz / 16 + (nnz & 15), 1, 1), dim3(32, 4, 1)>>>(
//         k, nnz, rowind.data_ptr<int>(), colind.data_ptr<int>(),
//         D1.data_ptr<float>(), D2.data_ptr<float>(), out.data_ptr<float>());
//   }
//   return out;
// }
