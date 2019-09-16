#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_ARRAY(x) CHECK_INPUT(x); AT_ASSERTM(x.dim() == 1, #x "must be dim1")
#define CHECK_MATRIX(x) CHECK_INPUT(x); AT_ASSERTM(x.dim() == 2, #x "must be dim2")

/* Kernel-Wrappers Defined In .cu File */
void launchRepeatKernel(
    at::Tensor& repeat_indices,
    at::Tensor const& repeats,
    at::Tensor const& offsets
);

//void launchPointwiseSubKernel(
    //float        *const       d_convData,
    //size_t  const             ldConvData,
    //float   const*const*const d_tempDataPtrs,
    //size_t  const             ldTempData,
    //int64_t const*const*const d_tempIndPtrs,
    //int64_t const*const       d_spikeLookup,
    //size_t  const             nnzRows,
    //int64_t const*const       d_spikeTemps,
    //int64_t const*const       d_spikeTimes,
    //int64_t const*const       d_spikeRowOffset
//);

void launchSplineSubKernel(
    size_t  const             numCoefs,
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
    int64_t const*const       d_eventBlockOffset,
    float   const             d_tempScaling
);

void launchRefracFillKernel(
    size_t const              fill_length,
    size_t const              fill_offset,
    float const               fill_value,
    float        *const       d_convData,
    size_t  const             ldConvData,
    int64_t const*const       d_spikeTemps,
    int64_t const*const       d_spikeTimes,
    size_t  const             numSpikes
);

/* Data Structure For Row-Sparse (M,N) Matrix. 
 *  - data stored as dense (nnz_rows, N) matrix
 *  - sparisty pattern stored as an (nnz_rows,) index array
 */
template<typename data_t, typename index_t>
class RowSparseTensor
{
    protected:
        at::Tensor nzData;
        at::Tensor nzInd;

    public:
        RowSparseTensor(const at::Tensor& nzData,
                        const at::Tensor& nzInd)
            : nzData(nzData), nzInd(nzInd) 
        {
            CHECK_MATRIX(nzData);
            CHECK_ARRAY(nzInd);
            AT_ASSERTM(nzData.size(0) == nzInd.size(0),
                       "Number of nzInd must match number of nzData rows");
        }
        RowSparseTensor(const RowSparseTensor& existing)
            : nzData(existing.nzData), nzInd(existing.nzInd) 
        {}

        /* Wrap Relevant Dim/Size Information */
        size_t nnz() const { return nzData.size(0); }
        size_t length() const { return nzData.size(1); }
        size_t ld() const { return nzData.stride(0); }

        /* Expose Tensors For Use With Pytorch */
        const at::Tensor& data() const { return nzData; } 
        const at::Tensor& ind() const { return nzInd; } 
        
        /* Expose Raw Pointers For Use With Custom/3rd-Party Kernels */
        data_t const* dataPtr() const { return nzData.data<data_t>(); }
        index_t const* indPtr() const { return nzInd.data<index_t>(); }
};

/* Data Structure To Represent A 3D (MxNxT) Sparse Tensor of Time-series.
 * i.e. each MxN slice has an identical sparsity pattern such that each
 * (i,j,:) slice either indexes out a dense or zero vector.
 *  - data structure viewed as M-independent row-sparse (N,T) matrices
 *  - memory allocated on device to aggregate data & index ptrs
 *  - memory allocated on device to aggregate varying nnz_rows sizes
 */
template<typename data_t, typename index_t>
class NeuralTemplates
{
    protected:
        const size_t _ld;
        std::vector<RowSparseTensor<data_t, index_t>> rowSparseTemplates;
        at::Tensor nzRowCounts;
        data_t const** d_dataPtrs; 
        index_t const** d_indPtrs; 

    private:
        void initPtrPtrs()
        {
            // Compute Allocate Sizes
            const size_t szd = sizeof(data_t const*) * this->size();
            const size_t szi = sizeof(index_t const*) * this->size();

            // Allocate Persistent Device Ptr Arrays
            cudaMalloc((void***)&d_dataPtrs, szd);
            cudaMalloc((void***)&d_indPtrs, szi);

            // Allocate Temporary CPU Ptr Arrays
            data_t const** h_dataPtrs = (data_t const**) malloc(szd);
            index_t const** h_indPtrs = (index_t const**) malloc(szi);

            // Fill Cpu Ptr Arrays
            for (size_t tdx = 0; tdx < this->size(); tdx++){
                h_dataPtrs[tdx] = rowSparseTemplates[tdx].dataPtr();
                h_indPtrs[tdx] = rowSparseTemplates[tdx].indPtr();
            }

            // Copy Cpu Ptr Arrays To Device Ptr Arrays
            cudaMemcpy(d_dataPtrs, h_dataPtrs, szd, cudaMemcpyHostToDevice);
            cudaMemcpy(d_indPtrs, h_indPtrs, szi, cudaMemcpyHostToDevice);

            // Free Temporary Cpu Ptr Arrays
            free(h_dataPtrs);
            free(h_indPtrs);
        }

        void initCountTensor()
        {
            // Allocate Temporary Tensor On CPU
            auto cpu_sizes = at::empty({nzRowCounts.size(0)},
                                       nzRowCounts.options().device(torch::kCPU));
            auto acc = cpu_sizes.accessor<index_t, 1>();

            // Aggregate Sizes From Templates In CPU Tensor
            for (size_t tdx = 0; tdx < this->size(); tdx++)
                acc[tdx] = (index_t)rowSparseTemplates[tdx].nnz();

            // Copy CPU Tensor's Aggregated Data To Empty Device Tensor
            nzRowCounts.copy_(cpu_sizes);
        }

    public:
        /* Allocate Memory On Device To Store Pointers & NNZ-Row Counts */
        NeuralTemplates(std::vector<RowSparseTensor<data_t, index_t>>&& temps):
            _ld(temps[0].ld()), rowSparseTemplates(temps), 
            nzRowCounts(at::empty({(int)temps.size()}, temps[0].ind().options()))
        { 
            // TODO Validate All Templates Have Compatible Shape
            // Fill NNZ Count Tensor
            initCountTensor();
            // Put Pointers Onto The Device 
            initPtrPtrs();
        }

        /* Free Allocated Device Memory */
        virtual ~NeuralTemplates(){
            cudaFree(d_dataPtrs);
            cudaFree(d_indPtrs);
        }

        /* Wrap Relevant Dim/Size Information */
        size_t size() const { return rowSparseTemplates.size(); }
        size_t ld() const { return _ld; }

        /* Expose Tensors For Use With Pytorch */
        const at::Tensor& nnz() const { return nzRowCounts; }
        const RowSparseTensor<data_t, index_t>& operator[](size_t index) const {
            return rowSparseTemplates[index];
        }

        /* Expose Raw Pointers For Use With Custom/3rd-Party Kernels */
        data_t const*const* dataPtrs() const { return d_dataPtrs; }
        index_t const*const* indPtrs() const { return d_indPtrs; }
        index_t const* nnzPtr() const { return nzRowCounts.data<index_t>(); }
};

//* Performs Energy Subtraction Given Templates & Spike Id's + Times
 //*/
//void spikeSub(
        //at::Tensor                           & convData,
        //at::Tensor                      const& spikeTimes,
        //at::Tensor                      const& spikeTemps,
        //NeuralTemplates<float, int64_t> const& templates)
//{
    //// Validate Input Tensors
    //CHECK_MATRIX(convData);
    //CHECK_ARRAY(spikeTimes);
    //CHECK_ARRAY(spikeTemps);

    //// Compute Nonzero Row Info & Use To Construct RowId->SpikeId Lookup
    //auto spikeRowCounts = at::index_select(templates.nnz(), 0, spikeTemps);
    //auto spikeRowOffset = at::cumsum(spikeRowCounts, 0);
    //auto rowCount = spikeRowOffset[spikeRowOffset.size(0)-1];
    //auto spikeLookup = at::empty({rowCount.item<int64_t>()},
                                 //spikeRowCounts.options());
    //launchRepeatKernel(spikeLookup, spikeRowCounts, spikeRowOffset); 

    //// Perform Batched Template Subtraction 
    //launchPointwiseSubKernel(convData.data<float>(),
                             //convData.stride(0),
                             //templates.dataPtrs(),
                             //templates.ld(),
                             //templates.indPtrs(),
                             //spikeLookup.data<int64_t>(),
                             //spikeLookup.size(0),
                             //spikeTemps.data<int64_t>(),
                             //spikeTimes.data<int64_t>(),
                             //spikeRowOffset.data<int64_t>());
//}

/* Performs Energy Subtraction Given Templates & Spike Id's + Times
 */
void splineSub(
        at::Tensor                           & energyFunc,
        at::Tensor                      const& eventTimeIdx,
        at::Tensor                      const& eventTimeOffset,
        at::Tensor                      const& eventTempIds,
        NeuralTemplates<float, int64_t> const& tempCoefs,
        float                           const  tempScaling)

{
    // Validate Input Tensors
    CHECK_MATRIX(energyFunc);
    CHECK_ARRAY(eventTimeIdx);
    CHECK_ARRAY(eventTimeOffset);
    CHECK_ARRAY(eventTempIds);

    // Compute Nonzero Row Info & Use To Construct RowId->SpikeId Lookup
    auto eventBlockCounts = at::index_select(tempCoefs.nnz(), 0, eventTempIds);
    auto eventBlockOffset = at::cumsum(eventBlockCounts, 0);
    auto blockCount = eventBlockOffset[eventBlockOffset.size(0)-1];
    auto eventIdLookup = at::empty({blockCount.item<int64_t>()},
                                   eventBlockCounts.options());
    launchRepeatKernel(eventIdLookup, eventBlockCounts, eventBlockOffset); 

    // Perform Batched Template Subtraction 
    launchSplineSubKernel(
        tempCoefs[0].length(),
        energyFunc.data<float>(),
        energyFunc.stride(0),
        tempCoefs.dataPtrs(),
        tempCoefs.ld(),
        tempCoefs.indPtrs(),
        eventIdLookup.data<int64_t>(),
        eventIdLookup.size(0),
        eventTempIds.data<int64_t>(),
        eventTimeIdx.data<int64_t>(),
        eventTimeOffset.data<float>(),
        eventBlockOffset.data<int64_t>(),
        tempScaling);
}

/* Sets Energy Within Refractory Periods Given Spike Id's + Times
 */
void refracFill(
    at::Tensor      & convData,
    at::Tensor const& spikeTimes,
    at::Tensor const& spikeTemps,
    size_t     const  fillLength,
    size_t     const  fillOffset=0,
    float      const  fillValue=0.0)
                      
{
    // Validate Input Tensors
    CHECK_MATRIX(convData);
    CHECK_ARRAY(spikeTimes);
    CHECK_ARRAY(spikeTemps);
    
    // Perform Batched Fill
    launchRefracFillKernel(fillLength,
                           fillOffset,
                           fillValue,
                           convData.data<float>(),
                           convData.stride(0),
                           spikeTemps.data<int64_t>(),
                           spikeTimes.data<int64_t>(),
                           spikeTemps.size(0));
}

/* Module Bindings */
namespace py = pybind11;
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Individual Templates
    py::class_<RowSparseTensor<float, int64_t>>(m, "Template")
        .def(py::init<const at::Tensor&, const at::Tensor&>())
        .def("__len__", &RowSparseTensor<float, int64_t>::nnz)
        .def_property_readonly("data", &RowSparseTensor<float, int64_t>::data)
        .def_property_readonly("indices", &RowSparseTensor<float, int64_t>::ind);
    // Aggregated Templates 
    py::class_<NeuralTemplates<float, int64_t>>(m, "BatchedTemplates")
        .def(py::init<std::vector<RowSparseTensor<float, int64_t>>&&>())
        .def("__len__", &NeuralTemplates<float, int64_t>::size)
        .def("__getitem__", [](const NeuralTemplates<float, int64_t>& self,
                               size_t index) {
            if (index >= self.size()) throw py::index_error();
            return self[index];
        })
        .def_property_readonly("nnz", &NeuralTemplates<float, int64_t>::nnz);
    // Batched Template Subtraction 
    //m.def("subtract_spikes",
          //&spikeSub,
          //"Remove contributions to the energy function for a set of detected spikes.",
          //py::arg("energy"), 
          //py::arg("time_indices"), 
          //py::arg("time_offsets"), 
          //py::arg("templates")); 
          
    m.def("subtract_splines",
          &splineSub,
          "Remove contributions to the energy function for a set of detected spikes.",
          py::arg("energy"), 
          py::arg("times_indices"), 
          py::arg("times_offsets"), 
          py::arg("template_ids"), 
          py::arg("templates"),
          py::arg("tempScaling")); 
          
    m.def("refrac_fill",
          &refracFill,
          "Sets energy function during refractory period for a set of detected spikes.",
          py::arg("energy"), 
          py::arg("spike_times"), 
          py::arg("spike_ids"), 
          py::arg("fill_length"),
          py::arg("fill_offset")=0,
          py::arg("fill_value")=0.0); 
}
