#ifndef SPMM_SUPA
#define SPMM_SUPA
#include <supa.h>
#include <supa_runtime.h>

#include "supa_util.hpp"

template <typename Index, typename DType>
__global__ void
csrspmm_seqreduce_rowbalance_kernel(const Index nr, const Index feature_size,
                                    const Index rowPtr[], const Index colIdx[],
                                    const DType values[], const DType dnInput[],
                                    DType dnOutput[]) {
  Index row_tile = block_dim.y; // 8
  Index subwarp_id = thread_idx.y;
  Index stride = row_tile * grid_dim.x; // 8 * (m/8)
  Index row = block_idx.x * row_tile + subwarp_id;
  Index v_id = (block_idx.y * block_dim.x) + thread_idx.x;
  dnInput += v_id;
  dnOutput += v_id;
  E += v_id;
  DType val;
  // DType res = init(REDUCE::Op);
  Index col;
  for (; row < nr; row += stride) {
    DType res = 0;
    Index E_k_idx = -1;
    Index start = __ldg(rowPtr + row);
    Index end = __ldg(rowPtr + row + 1);
    if ((end - start) > 0) {
      for (Index p = start; p < end; p++) {
        DType val_pre_red;
        col = __ldg(colIdx + p);
        val = __guard_load_default_one<DType>(values, p);
        val_pre_red = val * __ldg(dnInput + col * feature_size);
        // if ((REDUCE::Op == MAX && (res < val_pre_red)) ||
        //     ((REDUCE::Op == MIN) && (res > val_pre_red))) {
        //   E_k_idx = col;
        // }
        res = res + val_pre_red;

        // res += val * __ldg(dnInput + col * feature_size);
      }
      // if (REDUCE::Op == MEAN) {
      //   res /= (end - start);
      // }
    } else {
      res = 0;
    }
    dnOutput[row * feature_size] = res;
    // E[row * feature_size] = E_k_idx;
  }
}

template <typename Index, typename DType>
__global__ void csrspmm_seqreduce_nnzbalance_kernel(
    const Index nr, const Index feature_size, const Index nnz_,
    const Index rowPtr[], const Index colIdx[], const DType values[],
    const DType dnInput[], DType dnOutput[]) {
  Index nnz = nnz_;
  if (nnz < 0)
    nnz = rowPtr[nr];

  Index Nnzdim_thread = block_dim.y * grid_dim.x;
  Index NE_PER_THREAD = CEIL(nnz, Nnzdim_thread);
  Index eid = (block_idx.x * block_dim.y + thread_idx.y) * NE_PER_THREAD;
  Index v_id = (block_idx.y * block_dim.x) + thread_idx.x;
  Index col = 0;
  DType val = 0.0;
  if (v_id < feature_size) {
    if (eid < nnz) {
      Index row = binary_search_segment_number<Index>(rowPtr, nr, nnz, eid);
      Index step = __ldg(rowPtr + row + 1) - eid;

      for (Index ii = 0; ii < NE_PER_THREAD; ii++) {
        if (eid >= nnz)
          break;
        if (ii < step) {
          col = __ldg(colIdx + eid) * feature_size;
          val += __guard_load_default_one<DType>(values, eid) *
                 __ldg(dnInput + col + v_id);

          eid++;
        } else {
          atomicAdd(&dnOutput[row * feature_size + v_id], val);

          row = binary_search_segment_number<Index>(rowPtr, nr, nnz, eid);
          step = __ldg(rowPtr + row + 1) - eid;
          col = __ldg(colIdx + eid) * feature_size;
          val = __guard_load_default_one<DType>(values, eid) *
                __ldg(dnInput + col + v_id);

          eid++;
        }
      }
      atomicAdd(&dnOutput[row * feature_size + v_id], val);
    }
  }
}

template <typename Index, typename DType, typename access_t>
__global__ void
csrspmm_parreduce_rowbalance_kernel(const Index nr, const Index feature_size,
                                    const Index rowPtr[], const Index colIdx[],
                                    const DType values[], const DType dnInput[],
                                    DType dnOutput[]) {
  constexpr Index CoarsenFactor = sizeof(access_t) / sizeof(DType);

  Index lane_id = (thread_idx.x & (32 - 1));
  Index stride = grid_dim.x * block_dim.y;
  Index row = block_idx.x * block_dim.y + thread_idx.y;

  // get the dense column offset
  Index col_offset = block_idx.y * 32 + (thread_idx.x >> 5) * CoarsenFactor;
  const DType *B_panel = dnInput + col_offset;
  DType *C_panel = dnOutput + col_offset;
  Index ldB = feature_size;
  Index ldC = feature_size;

  if (col_offset >= feature_size)
    return;
  if (col_offset + CoarsenFactor >= feature_size)
    goto Ndim_Residue;

  for (; row < nr; row += stride) {
    // declare accumulators
    DType c[CoarsenFactor];
#pragma unroll
    for (Index j = 0; j < CoarsenFactor; j++) {
      c[j] = 0;
    }
    DType buffer[CoarsenFactor];

    Index start = rowPtr[row];
    Index end = rowPtr[row + 1];
    Index k;
    DType val;
    DType val_pre_red;
    Index E_k_idx[CoarsenFactor] = {0};
    // DType res = init(REDUCE::Op);

    for (Index jj = start + lane_id; jj < end; jj += 32) {
      k = colIdx[jj];
      val = __guard_load_default_one<DType>(values, jj);

      // load B-elements in vector-type
      *(access_t *)buffer = *(access_t *)(B_panel + k * ldB);

#pragma unroll
      for (Index i = 0; i < CoarsenFactor; i++) {
        val_pre_red = val * buffer[i];
        // if ((REDUCE::Op == REDUCEOP::MAX && val_pre_red > c[i]) ||
        //     (REDUCE::Op == REDUCEOP::MIN && val_pre_red < c[i])) {
        //   c[i] = val_pre_red;
        //   E_k_idx[i] = k;
        // } else if (REDUCE::Op == REDUCEOP::SUM ||
        //            REDUCE::Op == REDUCEOP::MEAN) {
        //   c[i] += val_pre_red;
        // }
        c[i] += val_pre_red;
      }
    }

#pragma unroll
    for (Index i = 0; i < CoarsenFactor; i++) {
      DType temp_c = c[i];
      // row-wise reduction is a simple merge-tree
      SHFL_DOWN_REDUCE(c[i])
    }

    // store to C in vector-type
    if (lane_id == 0) {
      *(access_t *)(C_panel + row * ldC) = *(access_t *)c;
      // *(access_t *)(E + col_offset + row * ldC) = *(access_t *)E_k_idx;
    }
  }
  return;

Ndim_Residue:
  Index valid_lane_num = feature_size - col_offset;

  for (; row < nr; row += stride) {
    // get row offsets
    DType c[CoarsenFactor];
#pragma unroll
    for (Index j = 0; j < CoarsenFactor; j++) {
      c[j] = 0;
    }
    DType buffer[CoarsenFactor];
    // access_t res = init_zeros<access_t>();

    Index start = rowPtr[row];
    Index end = rowPtr[row + 1];
    Index k;
    DType val;
    DType val_pre_red;
    Index E_k_idx[CoarsenFactor] = {0};

    for (Index jj = start + lane_id; jj < end; jj += 32) {
      k = colIdx[jj];
      val = __guard_load_default_one<DType>(values, jj);

#pragma unroll
      for (Index i = 0; i < CoarsenFactor; i++) {
        if (i < valid_lane_num) {
          buffer[i] = B_panel[k * ldB + i];
        }
      }

#pragma unroll
      for (Index i = 0; i < CoarsenFactor; i++) {
        val_pre_red = val * buffer[i];
        // if ((REDUCE::Op == REDUCEOP::MAX && val_pre_red > c[i]) ||
        //     (REDUCE::Op == REDUCEOP::MIN && val_pre_red < c[i])) {
        //   c[i] = val_pre_red;
        //   E_k_idx[i] = k;
        // } else if (REDUCE::Op == REDUCEOP::SUM ||
        //            REDUCE::Op == REDUCEOP::MEAN) {
        //   c[i] += val_pre_red;
        // }
        c[i] += val_pre_red;
      }
    }

#pragma unroll
    for (Index i = 0; i < CoarsenFactor; i++) {
      DType temp_c = c[i];
      // row-wise reduction is a simple merge-tree
      SHFL_DOWN_REDUCE(c[i])
    }

    if (lane_id == 0) {
#pragma unroll
      for (Index i = 0; i < CoarsenFactor; i++) {
        if (i < valid_lane_num) {
          C_panel[row * ldC + i] = c[i];
          // E[col_offset + row * ldC + i] = E_k_idx[i];
        }
      }
    }
  }
}

template <typename Index, typename DType, typename access_t>
__global__ void
csrspmm_parreduce_nnzbalance_kernel(const Index nr, const Index feature_size,
                                    const Index nnz_, const Index rowPtr[],
                                    const Index colIdx[], const DType values[],
                                    const DType dnInput[], DType dnOutput[]) {
  constexpr Index CoarsenFactor = sizeof(access_t) / sizeof(DType);
  Index nnz = nnz_;
  if (nnz < 0)
    nnz = rowPtr[nr];

  Index lane_id = (thread_idx.x & (32 - 1));
  Index Nnzdim_warp_id = block_idx.x * block_dim.y + thread_idx.y;
  Index nz_start = Nnzdim_warp_id * 32;
  Index stride = grid_dim.x * (block_dim.y * 32);

  // get the dense column offset
  Index col_offset = block_idx.y * 32 + (thread_idx.x >> 5) * CoarsenFactor;
  const DType *B_panel = dnInput + col_offset;
  DType *C_panel = dnOutput + col_offset;
  Index ldB = feature_size;
  Index ldC = feature_size;

  Index k;
  DType v;
  DType c[CoarsenFactor] = {0};
  DType buffer[CoarsenFactor] = {0};

  if (col_offset >= feature_size)
    return;
  if (col_offset + CoarsenFactor >= feature_size)
    goto Ndim_Residue;

  for (int nz_id = nz_start + lane_id;
       nz_id < nnz + lane_id; // make sure NO warp loop-divergence
       nz_id += stride) {
    Index row = binary_search_segment_number<Index>(rowPtr, nr, nnz, nz_id);

    if (nz_id < nnz) {
      k = colIdx[nz_id];
      v = __guard_load_default_one<DType>(values, nz_id);
    } else {
      k = 0;
      v = 0.0f;
    }

    // load B-elements in vector-type
    *(access_t *)buffer = *(access_t *)(B_panel + k * ldB);
#pragma unroll
    for (int i = 0; i < CoarsenFactor; i++) {
      c[i] = buffer[i] * v;
    }

    // reduction
    Index row_intv =
        __shfl_sync(FULLMASK, row, (32 - 1)) - __shfl_sync(FULLMASK, row, 0);
    if (row_intv == 0) {
// if all non-zeros in this warp belong to the same row, use a simple reduction
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        // SHFL_DOWN_REDUCE(c[i]);
      }
      if (lane_id == 0) {
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          atomicAdd(C_panel + row * ldC + i, c[i]);
        }
      }
    } else {
      // if non-zeros belong to different rows, use a parallel-scan primitive
      // thread that holds the start of each segment are responsible for writing
      // results
      bool is_seg_start =
          ((__shfl_up_sync(FULLMASK, row, 1) != row) || (lane_id == 0));
      DType tmpv;
      Index tmpr;
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        SEG_SHFL_SCAN(c[i], tmpv, row, tmpr);
      }
      if (is_seg_start) {
// atomic add has no vector-type form.
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          atomicAdd(C_panel + row * ldC + i, c[i]);
        }
      }
    }
  }
  return;
Ndim_Residue:
  Index valid_lane_num = feature_size - col_offset;

  for (int nz_id = nz_start + lane_id;
       nz_id < nnz + lane_id; // make sure NO warp loop-divergence
       nz_id += stride) {
    Index row = binary_search_segment_number<Index>(rowPtr, nr, nnz, nz_id);

    if (nz_id < nnz) {
      k = colIdx[nz_id];
      v = __guard_load_default_one<DType>(values, nz_id);
    } else {
      k = 0;
      v = 0.0f;
    }

#pragma unroll
    for (int i = 0; i < CoarsenFactor; i++) {
      if (i < valid_lane_num) {
        c[i] = B_panel[k * ldB + i] * v;
      }
    }

    // reduction
    Index row_intv =
        __shfl_sync(FULLMASK, row, (32 - 1)) - __shfl_sync(FULLMASK, row, 0);
    if (row_intv == 0) {
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        // SHFL_DOWN_REDUCE(c[i]);
      }
      if (lane_id == 0) {
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          if (i < valid_lane_num) {
            atomicAdd(C_panel + row * ldC + i, c[i]);
          }
        }
      }
    } else {
      bool is_seg_start =
          ((__shfl_up_sync(FULLMASK, row, 1) != row) || (lane_id == 0));
      DType tmpv;
      Index tmpr;
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        SEG_SHFL_SCAN(c[i], tmpv, row, tmpr);
      }
      if (is_seg_start) {
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          if (i < valid_lane_num) {
            atomicAdd(C_panel + row * ldC + i, c[i]);
          }
        }
      }
    }
  }
  return;
}

#endif