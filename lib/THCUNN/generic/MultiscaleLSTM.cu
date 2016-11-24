#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/MultiscaleLSTM.cu"
#else

#include "../common.h"

void THNN_(MultiscaleLSTM_updateOutput)(
          THCState *state,
          // Inputs
          THCTensor *input,
          THCudaIntTensor *targets,
          THCudaIntTensor *batches,
          THCudaIntTensor *origins,
          // Outputs
          THCTensor *output,
          THCTensor *cellOutput,
          // Parameters
          THCTensor *inputWeight,
          THCTensor *recurrentWeight,
          THCTensor *bias,
          // Buffers
          THCudaIntTensor *numOutArcs  // Per time step
          THCudaIntTensor *numInArcs,  // Per time step and batch
          THCTensor *xW,
          THCTensor *hR,
          THCTensor *gates,
          THCTensor *outputGates,
          // Config
          int batchSize)
{
  // TODO Assert everything on same GPU
  // TODO Check input sizes

  // Get sizes
  int seqLength = THCudaIntTensor_max(state, numOutArcs, 0) + 1;
  int totalInputs = THCTensor_(size)(state, input, 0);
  int inputSize = THCTensor_(size)(state, input, 1);
  int hiddenSize = THCTensor_(size)(state, recurrentWeight, 0);

  // Resize outputs
  THCTensor_(resize3d)(state, output, seqLength, batchSize, hiddenSize);
  THCTensor_(resize3d)(state, cellOutput, seqLength, batchSize, hiddenSize);

  // Resize buffers
  THCTensor_(resize2d)(state, xW, totalInputs, 4 * hiddenSize);
  THCTensor_(resize3d)(state, hR, seqLength - 1, batchSize, 4 * hiddenSize);
  THCTensor_(resize2d)(state, gates, totalInputs, 4 * hiddenSize);
  // NOTE Can technically be seqLength - 1
  THCTensor_(resize3d)(state, outputGates, seqLength, batchSize, hiddenSize);
  THCudaIntTensor_resize2d(state, numInArcs, seqLength - 1, batchSize);
  THCudaIntTensor_resize1d(state, numOutArcs, seqLength - 1);

  // Initial states are zero
  THCTensor_(zero)(state, output);
  THCTensor_(zero)(state, cellOutput);

  // For debugging
  THCTensor_(set3d)(state, output, 0, 0, 1, ScalarConvert<int,real>::to(1));
  THCTensor_(set3d)(state, output, 0, 1, 0, ScalarConvert<int,real>::to(2));
  THCTensor_(set3d)(state, output, 0, 1, 1, ScalarConvert<int,real>::to(3));

  // Accumulation tensors need to be set to 0 too
  THCTensor_(zero)(state, outputGates);
  THCudaIntTensor_zero(state, numInArcs);

  int nThreads = totalInputs;

  // Calculate numInArcs
  countArcs<<<GET_BLOCKS(nThreads), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
    batchSize, totalInputs,
    THCudaIntTensor_data(state, targets),
    THCudaIntTensor_data(state, targetBatches),
    THCudaIntTensor_data(state, targetSteps),
    THCudaIntTensor_data(state, numInArcs),
    THCudaIntTensor_data(state, numOutArcs)
  );

  // Transform the input data
  #ifdef THC_REAL_IS_FLOAT
  THCudaBlas_Sgemm(
  #elif defined(THC_REAL_IS_HALF)
  THCudaBlas_Hgemm(
  #elif defined(THC_REAL_IS_DOUBLE)
  THCudaBlas_Dgemm(
  #endif
    state,
    'n', 'n',
    4 * hiddenSize,
    totalInputs,
    inputSize,
    ScalarConvert<int, real>::to(1),
    THCTensor_(data)(state, inputWeight),
    4 * hiddenSize,
    THCTensor_(data)(state, input),
    inputSize,
    ScalarConvert<int, real>::to(0),
    THCTensor_(data)(state, xW),
    4 * hiddenSize
  );

  // Create tensors to hold the slices at each step
  THCTensor *output_t = THCTensor_(new)(state);
  THCTensor *hR_t = THCTensor_(new)(state);
  THCTensor *xW_t = THCTensor_(new)(state);
  THCTensor *gates_t = THCTensor_(new)(state);
  THCudaIntTensor *targets_t = THCudaIntTensor_new(state);
  THCudaIntTensor *targetBatches_t = THCudaIntTensor_new(state);
  THCTensor *cellOutput_t = THCTensor_(new)(state);
  THCTensor *outputGates_t = THCTensor_(new)(state);
  THCudaIntTensor *numInArcs_t = THCudaIntTensor_new(state);

  int inputsSeen = 0;
  for (int t = 0; t < seqLength - 1; t++) {
    // Transform the previous hidden state
    THCTensor_(select)(state, output_t, output, 0, t);
    THCTensor_(select)(state, hR_t, hR, 0, t);
    #ifdef THC_REAL_IS_FLOAT
    THCudaBlas_Sgemm(
    #elif defined(THC_REAL_IS_HALF)
    THCudaBlas_Hgemm(
    #elif defined(THC_REAL_IS_DOUBLE)
    THCudaBlas_Dgemm(
    #endif
      state,
      'n', 'n',
      hiddenSize * 4,
      batchSize,
      hiddenSize,
      ScalarConvert<int, real>::to(1),
      THCTensor_(data)(state, recurrentWeight),
      hiddenSize * 4,
      THCTensor_(data)(state, output_t),
      hiddenSize,
      ScalarConvert<int, real>::to(0),
      THCTensor_(data)(state, hR_t),
      hiddenSize * 4
    );

    // Perform the LSTM transitions
    int numOutArcs_t = THCudaIntTensor_get1d(state, numOutArcs, t);

    THCTensor_(narrow)(state, xW_t, xW, 0, inputsSeen, numOutArcs_t);
    THCTensor_(narrow)(state, gates_t, gates, 0, inputsSeen, numOutArcs_t);
    THCudaIntTensor_narrow(state, targets_t, targets, 0, inputsSeen, numOutArcs_t);
    THCudaIntTensor_narrow(state, targetBatches_t, targetBatches, 0, inputsSeen, numOutArcs_t);

    inputsSeen += numOutArcs_t;

    int nThreads = numOutArcs_t * hiddenSize;

    lstmElemwise<real><<<GET_BLOCKS(nThreads), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
        t, hiddenSize, batchSize,
        THCudaIntTensor_data(state, targets_t),
        THCudaIntTensor_data(state, targetBatches_t),
        numOutArcs_t,
        THCTensor_(data)(state, hR_t),
        THCTensor_(data)(state, xW_t),
        THCTensor_(data)(state, bias),
        THCTensor_(data)(state, gates_t),
        THCTensor_(data)(state, cellOutput),
        THCTensor_(data)(state, outputGates)
    );

    // Average the states of the next time step
    THCTensor_(select)(state, cellOutput_t, cellOutput, 0, t + 1);
    THCTensor_(select)(state, outputGates_t, outputGates, 0, t + 1);
    THCudaIntTensor_select(state, numInArcs_t, numInArcs, 0, t);
    THCTensor_(select)(state, output_t, output, 0, t + 1);

    nThreads = batchSize * hiddenSize;

    calculateState<<<GET_BLOCKS(nThreads), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
      hiddenSize, batchSize,
      THCTensor_(data)(state, cellOutput_t),
      THCTensor_(data)(state, outputGates_t),
      THCudaIntTensor_data(state, numInArcs_t),
      THCTensor_(data)(state, output_t)
    );
  }

}

void THNN_(MultiscaleLSTM_updateGradInput)(
          THCState *state,
          // Inputs
          THCTensor *input,
          THCTensor *gradInput,
          THCudaIntTensor *targets,
          THCudaIntTensor *targetBatches,
          // Outputs
          THCTensor *output,
          THCTensor *gradOutput,
          THCTensor *cellOutput,
          THCTensor *gradCellOutput,
          // Parameters
          THCTensor *inputWeight,
          THCTensor *recurrentWeight,
          THCTensor *bias,
          // Buffers
          THCudaIntTensor *numInArcs,
          THCudaIntTensor *numOutArcs,
          THCTensor *xW,
          THCTensor *hR,
          THCTensor *gates,
          THCTensor *gradGates,
          THCTensor *outputGates,
          THCTensor *gradOutputGates,
          int batchSize)
{
  // Get sizes
  int seqLength = THCudaIntTensor_size(state, numOutArcs, 0) + 1;
  int totalInputs = THCTensor_(size)(state, input, 0);
  int inputSize = THCTensor_(size)(state, input, 1);
  int hiddenSize = THCTensor_(size)(state, recurrentWeight, 0);

  // Resize buffers
  THCTensor_(resizeAs)(state, gradInput, input);
  THCTensor_(resizeAs)(state, gradGates, gates);
  THCTensor_(resizeAs)(state, gradCellOutput, cellOutput);
  THCTensor_(resizeAs)(state, gradOutputGates, outputGates);

  // Create tensors to view slices
  THCTensor *gradOutput_t = THCTensor_(new)(state);
  THCTensor *cellOutput_t = THCTensor_(new)(state);
  THCTensor *gradCellOutput_t = THCTensor_(new)(state);
  THCTensor *outputGates_t = THCTensor_(new)(state);
  THCTensor *gradOutputGates_t = THCTensor_(new)(state);
  THCudaIntTensor *numInArcs_t = THCudaIntTensor_new(state);

  int* arcCount;

  // Find the origins of each arc
  int* offsets;
  int* arcCount;
  int n = batchSize * (seqLength - 1);
  cudaMalloc(&offsets, n * sizeof(int));
  cudaMalloc(&arcCount, n * sizeof(int));
  thrust::device_ptr<int> offsets_ptr(offsets);
  thrust::device_ptr<int> numInArcs_ptr(THCudaIntTensor_data(state, numInArcs));
  thrust::exclusive_scan(numInArcs_ptr, numInArcs_ptr + n, offsets_ptr);

  int nThreads = totalInputs;

  reverseArcs<<<GET_BLOCKS(nThreads), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
    batchSize, totalInputs, seqLength,
    THCudaIntTensor_data(state, targets),
    THCudaIntTensor_data(state, targetBatches),
    THCudaIntTensor_data(state, origins),
    THCudaIntTensor_data(state, originBatches),
    offsets, arcCount
  );

  cudaFree(offsets);
  THCudaCheck(cudaGetLastError());

  for (int t = seqLength - 1; t > 0; t--) {
    THCTensor_(select)(state, cellOutput_t, cellOutput, 0, t);
    THCTensor_(select)(state, gradCellOutput_t, gradCellOutput, 0, t);
    THCTensor_(select)(state, outputGates_t, outputGates, 0, t);
    THCTensor_(select)(state, gradOutputGates_t, gradOutputGates, 0, t);
    THCudaIntTensor_select(state, numInArcs_t, numInArcs, 0, t - 1);
    THCTensor_(select)(state, gradOutput_t, gradOutput, 0, t);

    nThreads = batchSize * hiddenSize;

    calculateGradState<real><<<GET_BLOCKS(nThreads), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
      hiddenSize, batchSize,
      THCTensor_(data)(state, cellOutput_t),
      THCTensor_(data)(state, gradCellOutput_t),
      THCTensor_(data)(state, outputGates_t),
      THCTensor_(data)(state, gradOutputGates_t),
      THCudaIntTensor_data(state, numInArcs_t),
      THCTensor_(data)(state, gradOutput_t)
    );
  }
}

#endif
