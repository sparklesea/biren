#ifndef DATA_LOADER
#define DATA_LOADER

#include "supa_util.hpp"
#include "ramArray.hpp"
#include "mmio.hpp"
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <tuple>
#include <typeinfo>
#include <vector>

void read_mtx_file(const char *filename, int &nrow, int &ncol, int &nnz,
                   std::vector<int> &csr_indptr_buffer,
                   std::vector<int> &csr_indices_buffer) {
  FILE *f;

  if ((f = fopen(filename, "r")) == NULL) {
    printf("File %s not found", filename);
    exit(EXIT_FAILURE);
  }

  MM_typecode matcode;
  // Read MTX banner
  if (mm_read_banner(f, &matcode) != 0) {
    printf("Could not process this file.\n");
    exit(EXIT_FAILURE);
  }
  if (mm_read_mtx_crd_size(f, &nrow, &ncol, &nnz) != 0) {
    printf("Could not process this file.\n");
    exit(EXIT_FAILURE);
  }
  // printf("Reading matrix %d rows, %d columns, %d nnz.\n", nrow, ncol, nnz);

  /// read tuples

  std::vector<std::tuple<int, int>> coords;
  int row_id, col_id;
  float dummy;
  for (int64_t i = 0; i < nnz; i++) {
    if (fscanf(f, "%d", &row_id) == EOF) {
      std::cout << "Error: not enough rows in mtx file.\n";
      exit(EXIT_FAILURE);
    } else {
      fscanf(f, "%d", &col_id);
      if (mm_is_integer(matcode) || mm_is_real(matcode)) {
        fscanf(f, "%f", &dummy);
      }
      // mtx format is 1-based
      coords.push_back(std::make_tuple(row_id - 1, col_id - 1));
    }
  }

  if (mm_is_symmetric(matcode)) {
    std::vector<std::tuple<int, int>> new_coords;
    for (auto iter = coords.begin(); iter != coords.end(); iter++) {
      int i = std::get<0>(*iter);
      int j = std::get<1>(*iter);
      if (i != j) {
        new_coords.push_back(std::make_tuple(i, j));
        new_coords.push_back(std::make_tuple(j, i));
        nnz += 1;
      } else
        new_coords.push_back(std::make_tuple(i, j));
    }
    std::sort(new_coords.begin(), new_coords.end());
    coords.clear();
    for (auto iter = new_coords.begin(); iter != new_coords.end(); iter++) {
      if ((iter + 1) == new_coords.end() || (*iter != *(iter + 1))) {
        coords.push_back(*iter);
      }
    }
  } else {
    std::sort(coords.begin(), coords.end());
  }

  /// generate csr from coo
  csr_indptr_buffer.clear();
  csr_indices_buffer.clear();

  int curr_pos = 0;
  csr_indptr_buffer.push_back(0);
  for (int64_t row = 0; row < nrow; row++) {
    while ((curr_pos < nnz) && (std::get<0>(coords[curr_pos]) == row)) {
      csr_indices_buffer.push_back(std::get<1>(coords[curr_pos]));
      // printf("row, col = %d, %d\n", row, std::get<1>(coords[curr_pos]));
      curr_pos++;
    }
    // assert((std::get<0>(coords[curr_pos]) > row || curr_pos == nnz));
    csr_indptr_buffer.push_back(curr_pos);
  }
  nnz = csr_indices_buffer.size();
}

template <class Index, class DType> struct SpMatCsrDescr_t {
  SpMatCsrDescr_t(int ncol_, std::vector<Index> &indptr,
                  std::vector<Index> &indices) {
    nrow = indptr.size() - 1;
    ncol = ncol_;
    nnz = indices.size();
    sp_csrptr.create(nrow + 1, indptr);
    sp_csrind.create(nnz, indices);
    sp_data.create(nnz);
    sp_data.fill_default_one();
  }
  void upload() {
    sp_csrptr.upload();
    sp_csrind.upload();
    sp_data.upload();
  }
  int nrow;
  int ncol;
  int nnz;
  util::RamArray<Index> sp_csrptr;
  util::RamArray<Index> sp_csrind;
  util::RamArray<DType> sp_data;
};

template <class Index, class DType>
SpMatCsrDescr_t<Index, DType> DataLoader(const char *filename) {
  int H_nrow, H_ncol, H_nnz;
  std::vector<Index> H_csrptr, H_csrind, H_coorow;
  read_mtx_file(filename, H_nrow, H_ncol, H_nnz, H_csrptr, H_csrind);

  // printf("H.nrow %d H.ncol %d H_nnz %d\n", H_nrow, H_ncol, H_nnz);

  SpMatCsrDescr_t<Index, DType> H(H_ncol, H_csrptr, H_csrind);

  return H;
}

#endif