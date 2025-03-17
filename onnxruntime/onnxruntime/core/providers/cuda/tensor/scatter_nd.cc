// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tensor/scatter_nd.h"
#include "core/providers/cuda/tensor/scatter_nd_impl.h"
#include "core/providers/cuda/tensor/scatter_nd_common.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(ScatterND,
                                  kOnnxDomain,
                                  11, 12,
                                  kCudaExecutionProvider,
                                  (*KernelDefBuilder::Create())
                                      .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
                                      .MayInplace(0, 0),
                                  ScatterNDDisjointAndNoReduction);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(ScatterND,
                                  kOnnxDomain,
                                  13, 15,
                                  kCudaExecutionProvider,
                                  (*KernelDefBuilder::Create())
                                      .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
                                      .MayInplace(0, 0),
                                  ScatterNDWithAtomicReduction);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(ScatterND,
                                  kOnnxDomain,
                                  16, 17,
                                  kCudaExecutionProvider,
                                  (*KernelDefBuilder::Create())
                                      .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
                                      .MayInplace(0, 0),
                                  ScatterNDWithAtomicReduction);

ONNX_OPERATOR_KERNEL_EX(ScatterND,
                        kOnnxDomain,
                        18,
                        kCudaExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
                            .MayInplace(0, 0),
                        ScatterNDWithAtomicReduction);

static Status InitiliazeElementCountsAndInputDimsSpanOrGpu(int64_t last_index_dimension, const TensorShape& input_shape,
                                                           ElementCountsAndInputDimsSpanOrGpu& element_counts_and_input_dims,
                                                           CudaKernel::CudaAsyncBuffer<int64_t>& element_counts_and_input_dims_gpu,
                                                           onnxruntime::OpKernelContext* context) {
  TensorPitches input_strides(input_shape);

  if (last_index_dimension < 6) {
    element_counts_and_input_dims.gpu_ptr = nullptr;
    for (int64_t i = 0; i < last_index_dimension; ++i) {
      element_counts_and_input_dims.stack_ptr[i] = input_strides[i];
      element_counts_and_input_dims.stack_ptr[i + last_index_dimension] = input_shape[i];
    }
  } else {
    element_counts_and_input_dims_gpu.AllocCpuPtr(last_index_dimension * 2);
    memset(element_counts_and_input_dims_gpu.CpuPtr(), 0, sizeof(int64_t) * last_index_dimension * 2);
    for (int64_t i = 0; i < last_index_dimension; ++i) {
      element_counts_and_input_dims_gpu.CpuPtr()[i] = input_strides[i];
      element_counts_and_input_dims_gpu.CpuPtr()[i + last_index_dimension] = input_shape[i];
    }
    ORT_RETURN_IF_ERROR(element_counts_and_input_dims_gpu.CopyToGpu(context->GetComputeStream()));
    element_counts_and_input_dims.gpu_ptr = element_counts_and_input_dims_gpu.GpuPtr();
  }
  return Status::OK();
}

Status ScatterNDDisjointAndNoReduction::ComputeInternal(OpKernelContext* context) const {
  const auto* input_tensor = context->Input<Tensor>(0);
  const auto* indices_tensor = context->Input<Tensor>(1);
  const auto* updates_tensor = context->Input<Tensor>(2);

  const auto& input_shape = input_tensor->Shape();
  const auto& indices_shape = indices_tensor->Shape();
  const auto& updates_shape = updates_tensor->Shape();

  // Validate input shapes
  ORT_RETURN_IF_ERROR(onnxruntime::ScatterND::ValidateShapes(input_shape, indices_shape, updates_shape));

  auto* output_tensor = context->Output(0, input_shape);

  const void* input_data = input_tensor->DataRaw();
  void* output_data = output_tensor->MutableDataRaw();

  if (input_data != output_data) {
    // TODO: Run benchmarks to determine if a dedicated kernel doing data copy will be faster than invoking cudaMemcpy ?
    CUDA_RETURN_IF_ERROR(
        cudaMemcpyAsync(output_data, input_data, input_tensor->SizeInBytes(), cudaMemcpyDeviceToDevice, Stream(context)));
  }

  // Bail out early
  if (indices_shape.Size() == 0) {
    return Status::OK();
  }

  auto last_index_dimension = indices_shape[indices_shape.NumDimensions() - 1];
  size_t element_size = input_tensor->DataType()->Size();

  // We need element counts for each dimension and the input dim value for each dimension
  // for the range [0, last_index_dimension).
  // To avoid multiple GPU data transfers, we combine this into one array and send it through
  ElementCountsAndInputDimsSpanOrGpu element_counts_and_input_dims;
  CudaAsyncBuffer<int64_t> element_counts_and_input_dims_gpu(this);
  ORT_RETURN_IF_ERROR(InitiliazeElementCountsAndInputDimsSpanOrGpu(last_index_dimension, input_shape,
                                                                   element_counts_and_input_dims,
                                                                   element_counts_and_input_dims_gpu,
                                                                   context));

  ORT_RETURN_IF_ERROR(ScatterNDImpl(
      Stream(context),
      output_data,
      element_size,
      indices_shape.Size() / static_cast<size_t>(last_index_dimension),
      indices_tensor->Data<int64_t>(),  // only int64_t is supported for indices as per the onnx spec
      last_index_dimension,
      element_counts_and_input_dims,
      updates_tensor->DataRaw(),
      input_shape.SizeFromDimension(last_index_dimension)));

  return Status::OK();
}

Status ScatterNDWithAtomicReduction::ComputeInternal(OpKernelContext* context) const {
  const auto* input_tensor = context->Input<Tensor>(0);
  const auto* indices_tensor = context->Input<Tensor>(1);
  const auto* updates_tensor = context->Input<Tensor>(2);

  const auto& input_shape = input_tensor->Shape();
  const auto& indices_shape = indices_tensor->Shape();
  const auto& updates_shape = updates_tensor->Shape();

  // Validate input shapes
  ORT_RETURN_IF_ERROR(onnxruntime::ScatterND::ValidateShapes(input_shape, indices_shape, updates_shape));

  auto* output_tensor = context->Output(0, input_shape);

  const void* input_data = input_tensor->DataRaw();
  void* output_data = output_tensor->MutableDataRaw();

  if (input_data != output_data) {
    // TODO: Run benchmarks to determine if a dedicated kernel doing data copy will
    // be faster than invoking cudaMemcpy ?
    CUDA_RETURN_IF_ERROR(
        cudaMemcpyAsync(output_data, input_data, input_tensor->SizeInBytes(),
                        cudaMemcpyDeviceToDevice, Stream(context)));
  }

  // Bail out early
  if (indices_shape.Size() == 0) {
    return Status::OK();
  }

  auto last_index_dimension = indices_shape[indices_shape.NumDimensions() - 1];
  ElementCountsAndInputDimsSpanOrGpu element_counts_and_input_dims;
  CudaAsyncBuffer<int64_t> element_counts_and_input_dims_gpu(this);
  ORT_RETURN_IF_ERROR(InitiliazeElementCountsAndInputDimsSpanOrGpu(last_index_dimension, input_shape,
                                                                   element_counts_and_input_dims,
                                                                   element_counts_and_input_dims_gpu,
                                                                   context));

  switch (reduction_) {
    case ScatterNDReduction::None: {
      size_t element_size = input_tensor->DataType()->Size();
      ORT_RETURN_IF_ERROR(ScatterNDImpl(
          Stream(context),
          output_data,
          element_size,
          indices_shape.Size() / static_cast<size_t>(last_index_dimension),
          indices_tensor->Data<int64_t>(),  // only int64_t is supported for indices as per the onnx spec
          last_index_dimension,
          element_counts_and_input_dims,
          updates_tensor->DataRaw(),
          input_shape.SizeFromDimension(last_index_dimension)));
    } break;
    case ScatterNDReduction::Add:
    case ScatterNDReduction::Min:
    case ScatterNDReduction::Max:
    case ScatterNDReduction::Mul: {
      auto element_type = input_tensor->DataType()->AsPrimitiveDataType()->GetDataType();
      ORT_RETURN_IF_ERROR(ScatterNDImplReduction(
          Stream(context),
          output_data,
          element_type,
          indices_shape.Size() / static_cast<size_t>(last_index_dimension),
          indices_tensor->Data<int64_t>(),  // only int64_t is supported for indices as per the onnx spec
          last_index_dimension,
          element_counts_and_input_dims,
          updates_tensor->DataRaw(),
          input_shape.SizeFromDimension(last_index_dimension),
          reduction_));
    } break;
    default:
      ORT_THROW("ScatterND not supported for other reduction than Add, None.");
      break;
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
