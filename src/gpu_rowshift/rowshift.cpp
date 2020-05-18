#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_ARRAY(x) CHECK_INPUT(x); AT_ASSERTM(x.dim() == 1, #x "must be dim1")
#define CHECK_MATRIX(x) CHECK_INPUT(x); AT_ASSERTM(x.dim() == 2, #x "must be dim2")

/* Kernel-Wrappers Defined In .cu File */
void launchSignedShiftKernel(
    at::Tensor & matrix,
    at::Tensor const& shifts
);

void launchFwShiftKernel(
    at::Tensor & matrix,
    at::Tensor const& shifts
);

void launchBwShiftKernel(
    at::Tensor & matrix,
    at::Tensor const& shifts
);

/* 
 */
void signedShift(
    at::Tensor & matrix,
    at::Tensor const& shifts
){
    // Validate Input Tensors
    CHECK_MATRIX(matrix);
    CHECK_ARRAY(shifts);
    // TODO validate matrix.size(0) = shifts.size(0)
    // TODO validate shifts.stride(0) == 1

    // Perform Batched Template Subtraction 
    launchSignedShiftKernel(matrix, shifts);
}

/* 
 */
void fwShift(
    at::Tensor & matrix,
    at::Tensor const& shifts
){
    // Validate Input Tensors
    CHECK_MATRIX(matrix);
    CHECK_ARRAY(shifts);
    // TODO validate matrix.size(0) = shifts.size(0)
    // TODO validate shifts.stride(0) == 1

    // Perform Batched Template Subtraction 
    launchFwShiftKernel(matrix, shifts);
}

/* 
 */
void bwShift(
    at::Tensor & matrix,
    at::Tensor const& shifts
){
    // Validate Input Tensors
    CHECK_MATRIX(matrix);
    CHECK_ARRAY(shifts);
    // TODO validate matrix.size(0) = shifts.size(0)
    // TODO validate shifts.stride(0) == 1

    // Perform Batched Template Subtraction 
    launchBwShiftKernel(matrix, shifts);
}

/* Module Bindings */
namespace py = pybind11;
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("signed",
          &signedShift,
          "Inplace signed-shifting of matrix rows with 0-padding",
          py::arg("matrix"),
          py::arg("shifts"));  
    m.def("forward",
          &fwShift,
          "Inplace right-shifting of matrix rows with 0-padding",
          py::arg("matrix"),
          py::arg("shifts"));  
    m.def("backward",
          &bwShift,
          "Inplace left-shifting of matrix rows with 0-padding",
          py::arg("matrix"),
          py::arg("shifts"));  
}
