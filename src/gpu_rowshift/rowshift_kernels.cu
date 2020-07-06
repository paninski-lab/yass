#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define MAX_WARPS_PER_BLOCK 32

/* Utility: Offset Within C-Contiguous Matrix From Addr [0,0]
 */
template<typename data_t>
__device__ __forceinline__
data_t* pointer_offset(data_t * base_pointer,
                       dim3    const strides,
                       int const row,
                       int const col)
{
    return base_pointer + (strides.x * row + strides.y * col);
}

/* Each Block Is Assigned A Single Row (1D Grid Of 1D Blocks)
 */
template<typename data_t, typename index_t>
__global__ 
void fwShiftKernel(
    data_t       *const g_matrix,
    dim3    const       sizes,
    dim3    const       strides,
    index_t const*const g_shifts
){
    // Each Thread-Block Maps To Single Row, So Shift Can Be Shared
    __shared__ index_t s_shift;

    // Thread:0 Reads Global Shift Value To Shared Location
    if(threadIdx.x == 0){
        s_shift = g_shifts[blockIdx.x];
    }
    __syncthreads();

    // Broadcast Shared Shift Value To Thread-local Registers
    index_t const t_shift = s_shift;
    __syncthreads();

    // Slide Block-Sized Write Window From Right To Left
    if (t_shift > 0) {
        for (int pass = (sizes.y - 1) / blockDim.x; pass >= 0; pass--)
        {
            // Precompute Centered Locations & Inbounds Checks
            int const t_writeCol = (pass * blockDim.x) + threadIdx.x;
            data_t * const t_writeAddr = pointer_offset(
                    g_matrix, strides, blockIdx.x, t_writeCol
            );
            bool const write_inbounds = t_writeCol < sizes.y;
            bool const read_inbounds = write_inbounds && (t_writeCol >= t_shift);
    
            // Perform Reads From Left-Shifted Position
            data_t t_val = 0;
            if (read_inbounds){
                data_t * const t_readAddr = t_writeAddr - t_shift * strides.y;
                t_val = *t_readAddr;
            }
            __syncthreads();
        
            // Perform Writes To Centered Location
            if (write_inbounds){
                *t_writeAddr = t_val;
            }
            __syncthreads();
        }
    }
}

/* Each Block Is Assigned A Single Row (1D Grid Of 1D Blocks)
 */
template<typename data_t, typename index_t>
__global__ 
void bwShiftKernel(
    data_t       *const g_matrix,
    dim3    const       sizes,
    dim3    const       strides,
    index_t const*const g_shifts
){
    // Each Thread-Block Maps To Single Row, So Shift Can Be Shared
    __shared__ index_t s_shift;

    // Thread:0 Reads Global Shift Value To Shared Location
    if(threadIdx.x == 0){
        s_shift = g_shifts[blockIdx.x];
    }
    __syncthreads();

    // Broadcast Shared Shift Value To Thread-local Registers
    index_t const t_shift = s_shift;
    __syncthreads();

    // Slide Block-Sized Write Window From Left To Right
    if (t_shift > 0){
        for (int pass = 0; pass < (sizes.y - 1) / blockDim.x + 1; pass++)
        {
            // Precompute Centered Locations & Inbounds Checks
            int const t_writeCol = (pass * blockDim.x) + threadIdx.x;
            data_t * const t_writeAddr = pointer_offset(
                    g_matrix, strides, blockIdx.x, t_writeCol
            );
            bool const write_inbounds = t_writeCol < sizes.y;
            bool const read_inbounds = write_inbounds && (t_writeCol + t_shift < sizes.y);
    
            // Perform Reads From Right-Shifted Position
            data_t t_val = 0;
            if (read_inbounds){
                data_t * const t_readAddr = t_writeAddr + t_shift * strides.y;
                t_val = *t_readAddr;
            }
            __syncthreads();
        
            // Perform Writes To Centered Location
            if (write_inbounds){
                *t_writeAddr = t_val;
            }
            __syncthreads();
        }
    }
}

/* Each Block Is Assigned A Single Row (1D Grid Of 1D Blocks)
 */
template<typename data_t, typename index_t>
__global__ 
void signedShiftKernel(
    data_t       *const g_matrix,
    dim3    const       sizes,
    dim3    const       strides,
    index_t const*const g_shifts
){
    // Each Thread-Block Maps To Single Row, So Shift Can Be Shared
    __shared__ index_t s_shift;

    // Thread:0 Reads Global Shift Value To Shared Location
    if(threadIdx.x == 0){
        s_shift = g_shifts[blockIdx.x];
    }
    __syncthreads();

    // Broadcast Shared Shift Value To Thread-local Registers
    index_t const t_shift = s_shift;
    __syncthreads();
    if (t_shift >= 0) {

        // Slide Block-Sized Write Window From Right To Left
        for (int pass = (sizes.y - 1) / blockDim.x; pass >= 0; pass--)
        {
            // Precompute Centered Locations & Inbounds Checks
            int const t_writeCol = (pass * blockDim.x) + threadIdx.x;
            data_t * const t_writeAddr = pointer_offset(
                    g_matrix, strides, blockIdx.x, t_writeCol
            );
            bool const write_inbounds = t_writeCol < sizes.y;
            bool const read_inbounds = write_inbounds && (t_writeCol >= t_shift);
    
            // Perform Reads From Left-Shifted Position
            data_t t_val = 0;
            if (read_inbounds){
                data_t * const t_readAddr = t_writeAddr - t_shift * strides.y;
                t_val = *t_readAddr;
            }
            __syncthreads();
        
            // Perform Writes To Centered Location
            if (write_inbounds){
                *t_writeAddr = t_val;
            }
            __syncthreads();
        }
    } else if (t_shift < 0) {
        // Slide Block-Sized Write Window From Left To Right
        for (int pass = 0; pass < (sizes.y - 1) / blockDim.x + 1; pass++)
        {
            // Precompute Centered Locations & Inbounds Checks
            int const t_writeCol = (pass * blockDim.x) + threadIdx.x;
            data_t * const t_writeAddr = pointer_offset(
                    g_matrix, strides, blockIdx.x, t_writeCol
            );
            bool const write_inbounds = t_writeCol < sizes.y;
            bool const read_inbounds = write_inbounds && (t_writeCol - t_shift < sizes.y);
    
            // Perform Reads From Right-Shifted Position
            data_t t_val = 0;
            if (read_inbounds){
                data_t * const t_readAddr = t_writeAddr - t_shift * strides.y;
                t_val = *t_readAddr;
            }
            __syncthreads();
        
            // Perform Writes To Centered Location
            if (write_inbounds){
                *t_writeAddr = t_val;
            }
            __syncthreads();
        }
    }
}

/*
 */
void launchFwShiftKernel(
    at::Tensor & matrix,
    at::Tensor const& shifts
){
    // Determine Launch Configuration
    const int block = WARP_SIZE * MAX_WARPS_PER_BLOCK;
    const int grid = matrix.size(0);
    const dim3 sizes(matrix.size(0), matrix.size(1));
    const dim3 strides(matrix.stride(0), matrix.stride(1));

    // Dispatch Kernel 
    fwShiftKernel<float, int64_t><<<grid, block>>>(
        matrix.data<float>(),
        sizes,
        strides,
        shifts.data<int64_t>()
    );
}
/*
 */
void launchBwShiftKernel(
    at::Tensor & matrix,
    at::Tensor const& shifts
){
    // Determine Launch Configuration
    const int block = WARP_SIZE * MAX_WARPS_PER_BLOCK;
    const int grid = matrix.size(0);
    const dim3 sizes(matrix.size(0), matrix.size(1));
    const dim3 strides(matrix.stride(0), matrix.stride(1));

    // Dispatch Kernel 
    bwShiftKernel<float, int64_t><<<grid, block>>>(
        matrix.data<float>(),
        sizes,
        strides,
        shifts.data<int64_t>()
    );
}
/*
 */
void launchSignedShiftKernel(
    at::Tensor & matrix,
    at::Tensor const& shifts
){
    // Determine Launch Configuration
    // TODO: tune for optimal blocksizing
    const int block = WARP_SIZE * MAX_WARPS_PER_BLOCK;
    const int grid = matrix.size(0);
    const dim3 sizes(matrix.size(0), matrix.size(1));
    const dim3 strides(matrix.stride(0), matrix.stride(1));

    // Dispatch Kernel 
    signedShiftKernel<float, int64_t><<<grid, block>>>(
        matrix.data<float>(),
        sizes,
        strides,
        shifts.data<int64_t>()
    );
}
