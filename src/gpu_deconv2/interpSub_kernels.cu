#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define FULL_MASK 0xffffffff
#define WARP_SIZE 32

/* Utility: Compute Offset Within Matrix From Addr [0,0]
 */
template<typename data_t, typename index_t>
__device__ __forceinline__
data_t* pointerOffset(data_t * base_pointer,
                      size_t const ld,
                      index_t const row,
                      index_t const col)
{
    return base_pointer + (ld * row + col);
}

/* Utility: Nonnegative Integer Powers Of 2
 */
__device__ __forceinline__
int pow2(int exponent)
{
    int val = 1;
    for (int j = 0; j < exponent; j++) val *= 2;
    return val;
}

/* Utility: Value Of Coef Leading
 *    B_{j,m}(xi) : xi = ti + delta 
 */
template<typename data_t>
__device__ __forceinline__
data_t aCoef(int imj, int m, data_t delta){
    return (delta + (data_t)imj) / (data_t)m;
}

/* Utility: Value Of Coef Leading
 *    B_{j+1,m}(xi) : xi = ti + delta 
 */
template<typename data_t>
__device__ __forceinline__
data_t bCoef(int imj, int m, data_t delta){
    return ((data_t)m - (data_t)imj - delta) / (data_t)m;
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
void pointwiseSubKernel(
    data_t       *const       g_convData,
    size_t const              ldConvData,
    data_t  const*const*const g_tempDataPtrs,
    size_t const              ldTempData,
    index_t const*const*const g_tempIndPtrs,
    index_t const*const       g_spikeLookup,
    index_t const*const       g_spikeTemps,
    index_t const*const       g_spikeTimes,
    index_t const*const       g_spikeRowOffsets
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
        s_convDataRowHead = pointerOffset<data_t, index_t>(g_convData,
                                                            ldConvData,
                                                            g_tempIndPtrs[temp][tempRowId],
                                                            g_spikeTimes[spike]);
    
        // Compute Nonzero-Row Offset Within Template
        s_tempDataRowHead = pointerOffset<data_t const, index_t>(g_tempDataPtrs[temp],
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
        data_t const t_tempDataElem = *t_tempDataAddr * -2;
    
        // Write Results To Global Mem
        //__syncwarp(mask);
        atomicAdd(t_convDataAddr, t_tempDataElem);
    }
}

/* Each Block Is A Single Nonzero Row
 * (1D Grid Of 1D Blocks)
 * Subtract interpolated values from energy function after being given 
 * coefficients for cubic B-Spline and time offset:
 * Assumes:
 * Equidistant knots, constant offset, Boundary padding during fit.
 */
template<typename data_t, typename index_t, int order>
__global__ 
void splineSubKernel(
    size_t  const             numCoef,
    data_t       *const       g_energyVals,
    size_t  const             ldEnergyVals,
    data_t  const*const*const g_tempCoefPtrs,
    size_t  const             ldTempCoefs,
    index_t const*const*const g_tempIndPtrs,
    index_t const*const       g_eventIdLookup,
    index_t const*const       g_eventTempIds,
    index_t const*const       g_eventTimeIdx,
    data_t  const*const       g_eventTimeOffset,
    index_t const*const       g_eventBlockOffset
){
    // Basis Evals Same In Every Thread (Uniform Spacing) => Only Compute Once
    __shared__ data_t s_basisVals[order + 1];
               data_t t_basisVals[order + 1];

    // Each Coef Used In M Different Threads => Only Accesses Gmem Once
    extern __shared__ data_t s_tempCoefs[];

    // Each Block Matches Single Rows => Only Access/Compute Once
    __shared__ data_t const* s_tempCoefHeadAddr;
    __shared__ data_t      * s_energyValHeadAddr;
    __shared__ data_t        s_timeOffset;

    // Compute 2^order For Use In Basis Evaluation
    const int pow2order = pow2(order);

    // Initialize Evaled Basis Func Smem Buffer to 0
    if (threadIdx.x < order + 1){
        s_basisVals[threadIdx.x] = 0;
    }
    
    // tid0 Is Responsible For Computing Address Heads & Evluating Bases 
    if(threadIdx.x == 0)
    {
        // Lookup Spike's Template Row Offset
        const index_t eventId = g_eventIdLookup[blockIdx.x];
        const index_t tempId = g_eventTempIds[eventId];
        const index_t tempBlockId = (eventId > 0) ? 
            (blockIdx.x - g_eventBlockOffset[eventId-1]) : blockIdx.x;

        // Read Event Time Offsets & Adjust Time Idx 
        index_t t_eventTimeIdx = g_eventTimeIdx[eventId];
        data_t t_timeOffset = g_eventTimeOffset[eventId] * -1;
        if(t_timeOffset < 0){
            t_timeOffset += 1;
            t_eventTimeIdx += 1;
        }
        s_timeOffset = t_timeOffset;

        // Compute Address Of First Modified Value In Energy Function
        s_energyValHeadAddr = pointerOffset<data_t, index_t>(
            g_energyVals,
            ldEnergyVals,
            g_tempIndPtrs[tempId][tempBlockId],
            t_eventTimeIdx
        );
    
        // Compute Address Of First Spline Coefficient In Template 
        s_tempCoefHeadAddr = pointerOffset<data_t const, index_t>(
            g_tempCoefPtrs[tempId],
            ldTempCoefs,
            tempBlockId,
            0 // Assuming We Have Only Stored Interior Coeffs
        );
    }

    // Precompute Intermediate Coefficients & Atomic Add Bases
    __syncthreads();
    if (threadIdx.x < pow2order)
    {
        // Initialize Temporaru Quantities
        const data_t t_timeOffset = s_timeOffset;
        int id = threadIdx.x;
        int split = pow2order;
        int imj = 0;  // i - j : B_j,m(xi + delta)
        data_t coef = 1;

        // Compute Intermediate Coefficients
        for (int m = 1; m <= order; m++){
            split /= 2;
            if (id < split){
                coef *= bCoef<data_t>(imj, m, t_timeOffset);
                imj += 1;
            } else {
                coef *= aCoef<data_t>(imj, m, t_timeOffset);
            }
            id = id % split;
        }

        // Write Temporary Values To Buffer According To # Of Shifts
        atomicAdd(s_basisVals + (order - imj), coef);
    }

    // Copy Evaluted Basis From SMem Into Thread-local Registers
    __syncthreads();
    for (int j = 0; j < order + 1; j++){
        t_basisVals[j] = s_basisVals[j];
    }

    // In-Bounds Threads Perform Read/Write Of Template Coefs To Smem
    __syncthreads();
    data_t const*const t_tempCoefAddr = s_tempCoefHeadAddr+ threadIdx.x;
    if (threadIdx.x < numCoef){
        s_tempCoefs[threadIdx.x] = *t_tempCoefAddr;
    }

    // In-Bounds Threads Perform Reconstruction
    __syncthreads();
    data_t t_tempVal = 0;
    if (threadIdx.x < numCoef - order - 1){
        for (int j=0; j < order + 1; j++){
            t_tempVal -= s_tempCoefs[threadIdx.x + j] * t_basisVals[j];
        } 
    }

    // In-Bounds Threads Perform Subtraction 
    __syncthreads();
    data_t *const t_energyValAddr = s_energyValHeadAddr + threadIdx.x;
    if (threadIdx.x < numCoef - order - 1){
        atomicAdd(t_energyValAddr, 2 * t_tempVal);
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
        s_convDataRowHead = pointerOffset<data_t, index_t>(
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
    const int largestRow = at::max(repeats).item<int>();
    const int block = (largestRow > 288) ? (288) : largestRow;
    const dim3 grid((largestRow + 287) / 288, repeats.size(0));

    // Dispatch Kernel 
    repeatKernel<int64_t><<<grid, block>>>(repeats.data<int64_t>(),
                                           offsets.data<int64_t>(),
                                           repeat_indices.data<int64_t>());

    // TODO: Remove Cuda Error Checking For Performance
    cudaError_t errSync  = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess)
        printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
    if (errAsync != cudaSuccess)
        printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
}

/*
 */
void launchPointwiseSubKernel(
    float        *const       d_convData,
    size_t  const             ldConvData,
    float   const*const*const d_tempDataPtrs,
    size_t  const             ldTempData,
    int64_t const*const*const d_tempIndPtrs,
    int64_t const*const       d_spikeLookup,
    size_t  const             nnzRows,
    int64_t const*const       d_spikeTemps,
    int64_t const*const       d_spikeTimes,
    int64_t const*const       d_spikeRowOffset
){
    // Determine Launch Configuration
    const int block = ((ldTempData + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    const int grid = nnzRows;

    // Dispatch Kernel
    pointwiseSubKernel<float, int64_t><<<grid, block>>>(d_convData,
                                                       ldConvData,
                                                       d_tempDataPtrs,
                                                       ldTempData,
                                                       d_tempIndPtrs,
                                                       d_spikeLookup,
                                                       d_spikeTemps,
                                                       d_spikeTimes,
                                                       d_spikeRowOffset);

    // TODO: Remove Cuda Error Checking For Performance
    cudaError_t errSync  = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess)
        printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
    if (errAsync != cudaSuccess)
        printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
}

/*
 */
void launchSplineSubKernel(
    size_t  const             numCoef,
    float        *const       d_energyVals,
    size_t  const             ldEnergyVals,
    float   const*const*const d_tempCoefPtrs,
    size_t  const             ldTempCoefs,
    int64_t const*const*const d_tempIndPtrs,
    int64_t const*const       d_eventIdLookup,
    size_t  const             blockCount,
    int64_t const*const       d_eventTempIds,
    int64_t const*const       d_eventTimeIdx,
    float   const*const       d_eventTimeOffset,
    int64_t const*const       d_eventBlockOffset
){
    // Determine Launch Configuration
    const size_t ORDER = 3; // Hard Coded To Compile Cubic Bspline Kernel
    const int block = ((numCoef + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    const int grid = blockCount;
    const size_t smemAlloc = sizeof(float) * numCoef;

    // Dispatch Kernel
    splineSubKernel<float, int64_t, ORDER><<<grid, block, smemAlloc>>>(
        numCoef,
        d_energyVals,
        ldEnergyVals,
        d_tempCoefPtrs,
        ldTempCoefs,
        d_tempIndPtrs,
        d_eventIdLookup,
        d_eventTempIds,
        d_eventTimeIdx,
        d_eventTimeOffset,
        d_eventBlockOffset
    );

    // TODO: Remove Cuda Error Checking For Performance
    cudaError_t errSync  = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess)
        printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
    if (errAsync != cudaSuccess)
        printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
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

    // TODO: Remove Cuda Error Checking For Performance
    cudaError_t errSync  = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess)
        printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
    if (errAsync != cudaSuccess)
        printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
}
