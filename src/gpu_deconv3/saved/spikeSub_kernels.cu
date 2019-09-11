#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define FULL_MASK 0xffffffff
#define WARP_SIZE 32

/* Utility: Compute Offset Within Matrix From Addr [0,0]
 */
template<typename data_t, typename index_t>
__device__ __forceinline__
data_t* pointer_offset(data_t * base_pointer,
                       size_t const ld,
                       index_t const row,
                       index_t const col)
{
    return base_pointer + (ld * row + col);
}

/* Each Block Is (Potentially Subset Of) A Single Row 
 * (2D Grid Of 1D Blocks)
 */
// TODO restrict pointers for better compiler optim
template<typename index_t>
__global__ 
void repeatKernel(index_t const*const g_repeats,
                  index_t const*const g_offsets,
                  index_t      *const g_out)
{
    // Num Repeats & Offset Shared Across Entire Block 
    __shared__ int s_repeat;
    __shared__ int s_offset;

    // Thread ID 
    const int xid = blockIdx.x * blockDim.x + threadIdx.x;
    const int yid = blockIdx.y;

    // tid0 Responsible For Global -> Shared MeM XFER 
    if (threadIdx.x == 0){
        s_repeat = g_repeats[yid];
        s_offset = (yid > 0) ? g_offsets[yid - 1] : 0;
    }
    // In-Bounds Threads Write RowID As Output
    __syncthreads();
    if (xid < s_repeat)
        g_out[s_offset + xid] = yid;
}



/* Each Block Is A Single Nonzero Row
 * (1D Grid Of 1D Blocks)
 */
// TODO restrict pointers for better compiler optim
template<typename data_t, typename index_t>
__global__ 
void templateSubKernel(
    data_t       *const       g_convData,
    size_t const              ldConvData,
    data_t  const*const*const g_tempDataPtrs,
    size_t const              ldTempData,
    index_t const*const*const g_tempIndPtrs,
    index_t const*const       g_spikeLookup,
    index_t const*const       g_spikeTemps,
    index_t const*const       g_spikeTimes,
    index_t const*const       g_spikeRowOffsets,
    data_t  const             g_tempScale
){
    // Each Thread-Block Maps To One Row, So Row-Heads Can Be Shared
    __shared__ data_t      * s_convDataRowHead;
    __shared__ data_t const* s_tempDataRowHead;

    // tid0 Is Responsible For Computing Row-Heads 
    if(threadIdx.x == 0)
    {
        // Lookup Spike's Template Row Offset
        const index_t spike = g_spikeLookup[blockIdx.x];
        const index_t temp = g_spikeTemps[spike];
        const index_t tempRowId = (spike > 0) ? 
            (blockIdx.x - g_spikeRowOffsets[spike-1]) : blockIdx.x;

        // Compute Template & Time Offset In Convolved
        s_convDataRowHead = pointer_offset<data_t, index_t>(g_convData,
                                                            ldConvData,
                                                            g_tempIndPtrs[temp][tempRowId],
                                                            g_spikeTimes[spike]);
    
        // Compute Nonzero-Row Offset Within Template
        s_tempDataRowHead = pointer_offset<data_t const, index_t>(g_tempDataPtrs[temp],
                                                                  ldTempData,
                                                                  tempRowId,
                                                                  0);
    }
    // In-Bounds Threads Perform Indpendent Reads->Sub->Write
    //unsigned mask = __ballot_sync(FULL_MASK, threadIdx.x < ldTempData);
    __syncthreads();
    if (threadIdx.x < ldTempData)
    {
        // Compute Thread-Unique Addresses
        //__syncwarp(mask);
        data_t const*const t_tempDataAddr = s_tempDataRowHead + threadIdx.x;
        //__syncwarp(mask);
        data_t      *const t_convDataAddr = s_convDataRowHead + threadIdx.x;
    
        // Perform Global Mem Reads
        //__syncwarp(mask);
        // TODO handle scaling directly on templates? Alternatively as a param?
        data_t const t_tempDataElem = *t_tempDataAddr * -g_tempScale;
    
        // Write Results To Global Mem
        //__syncwarp(mask);
        atomicAdd(t_convDataAddr, t_tempDataElem);
    }
}

/* Each Block Is A Single Spike (i.e. row of convData) 
 * (1D Grid Of 1D Blocks)
 */
// TODO restrict pointers for better compiler optim
template<typename data_t, typename index_t>
__global__ 
void refracFillKernel(
    size_t const              fill_length,
    size_t const              fill_offset,
    data_t const              fill_value,
    data_t       *const       g_convData,
    size_t const              ldConvData,
    index_t const*const       g_spikeTemps,
    index_t const*const       g_spikeTimes
){
    // Get Addr For tid0 By Looking Up Spike Time(col) & Template(row)
    __shared__ data_t *s_convDataRowHead;
    if(threadIdx.x == 0){
        s_convDataRowHead = pointer_offset<data_t, index_t>(
            g_convData,
            ldConvData,
            g_spikeTemps[blockIdx.x],
            g_spikeTimes[blockIdx.x] + fill_offset
        );
    }
    // In-Bounds Threads Perform Indpendent Writes
    __syncthreads();
    if (threadIdx.x < fill_length){
        data_t *const t_convDataAddr = s_convDataRowHead + threadIdx.x;
        *t_convDataAddr = fill_value;
    }
}

/*
 */
void launchRepeatKernel(
    at::Tensor      & repeat_indices,
    at::Tensor const& repeats,
    at::Tensor const& offsets
){
    // Determine Launch Configuration
    // TODO experiment to find "optimal" block size, or make it a param
    const int largestRow = at::max(repeats).item<int>();
    const int block = (largestRow > 288) ? (288) : largestRow;
    const dim3 grid((largestRow + 287) / 288, repeats.size(0));

    // Dispatch Kernel 
    repeatKernel<int64_t><<<grid, block>>>(repeats.data<int64_t>(),
                                           offsets.data<int64_t>(),
                                           repeat_indices.data<int64_t>());
}

/*
 */
void launchTemplateSubKernel(
    float        *const       d_convData,
    size_t  const             ldConvData,
    float   const*const*const d_tempDataPtrs,
    size_t  const             ldTempData,
    int64_t const*const*const d_tempIndPtrs,
    int64_t const*const       d_spikeLookup,
    size_t  const             nnzRows,
    int64_t const*const       d_spikeTemps,
    int64_t const*const       d_spikeTimes,
    int64_t const*const       d_spikeRowOffset,
    float   const             d_tempScale
){
    // Determine Launch Configuration
    const int block = ((ldTempData + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    const int grid = nnzRows;

    // Dispatch Kernel
    templateSubKernel<float, int64_t><<<grid, block>>>(d_convData,
                                                       ldConvData,
                                                       d_tempDataPtrs,
                                                       ldTempData,
                                                       d_tempIndPtrs,
                                                       d_spikeLookup,
                                                       d_spikeTemps,
                                                       d_spikeTimes,
                                                       d_spikeRowOffset,
                                                       d_tempScale);
}

/*
 */
void launchRefracFillKernel(
    size_t  const             fill_length,
    size_t  const             fill_offset,
    float   const             fill_value,
    float        *const       d_convData,
    size_t  const             ldConvData,
    int64_t const*const       d_spikeTemps,
    int64_t const*const       d_spikeTimes,
    size_t  const             numSpikes
){
    // Determine Launch Configuration
    const int block = ((fill_length + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    const int grid = numSpikes;

    // Dispatch Kernel
    refracFillKernel<float, int64_t><<<grid, block>>>(fill_length,
                                                      fill_offset,
                                                      fill_value,
                                                      d_convData,
                                                      ldConvData,
                                                      d_spikeTemps,
                                                      d_spikeTimes);
}
