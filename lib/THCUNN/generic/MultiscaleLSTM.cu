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
          // Inputs
          THCTensor *output,
          THCTensor *cellOutput,
          // Parameters
          THCTensor *inputWeight,
          THCTensor *recurrentWeight,
          THCTensor *bias,
          // Buffers
          THCudaIntTensor *numOutArcs, // Per time step
          THCudaIntTensor *numInArcs,  // Per time step
          THCTensor *normalizingConstants,  // Incoming arcs per step and batch
          THCTensor *xW,
          THCTensor *hR,
          THCTensor *gates,
          THCTensor *outputGates,
          // Config
          int batchSize)
{
  // Get sizes
  int totalInputs = THCTensor_(size)(state, input, 0);
  int inputSize = THCTensor_(size)(state, input, 1);
  int hiddenSize = THCTensor_(size)(state, recurrentWeight, 0);

  // The sequence length is the number of hidden states, excluding the initial
  // This means it's equal to the number of elements in the sequence
  thrust::device_ptr<int> targets_ptr(THCudaIntTensor_data(state, targets));
  int seqLength = *thrust::max_element(targets_ptr, targets_ptr + totalInputs);

  // Resize outputs
  THCTensor_(resize3d)(state, output, seqLength + 1, batchSize, hiddenSize);
  THCTensor_(resize3d)(state, cellOutput, seqLength + 1, batchSize, hiddenSize);

  // Resize buffers
  THCTensor_(resize2d)(state, xW, totalInputs, 4 * hiddenSize);
  THCTensor_(resize2d)(state, gates, totalInputs, 4 * hiddenSize);

  THCTensor_(resize3d)(state, hR, seqLength, batchSize, 4 * hiddenSize);
  THCTensor_(resize3d)(state, outputGates, seqLength, batchSize, hiddenSize);

  THCTensor_(resize2d)(state, normalizingConstants, seqLength, batchSize);
  THCudaIntTensor_resize1d(state, numInArcs, seqLength);
  THCudaIntTensor_resize1d(state, numOutArcs, seqLength);

  // Initial states are zero
  THCTensor_(zero)(state, output);
  THCTensor_(zero)(state, cellOutput);

  // Accumulation tensors need to be set to 0 too
  THCTensor_(zero)(state, outputGates);
  THCTensor_(zero)(state, normalizingConstants);
  THCudaIntTensor_zero(state, numInArcs);
  THCudaIntTensor_zero(state, numOutArcs);

  // For debugging
  THCTensor_(set3d)(state, output, 0, 0, 1, ScalarConvert<int,real>::to(1));
  THCTensor_(set3d)(state, output, 0, 1, 0, ScalarConvert<int,real>::to(2));
  THCTensor_(set3d)(state, output, 0, 1, 1, ScalarConvert<int,real>::to(3));

  int nThreads = totalInputs;

  // Count the number of arcs going in and out at each step
  countArcs<real><<<GET_BLOCKS(nThreads), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
    batchSize, totalInputs,
    THCudaIntTensor_data(state, targets),
    THCudaIntTensor_data(state, batches),
    THCudaIntTensor_data(state, origins),
    THCudaIntTensor_data(state, numInArcs),
    THCudaIntTensor_data(state, numOutArcs),
    THCTensor_(data)(state, normalizingConstants)
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
  THCudaIntTensor *batches_t = THCudaIntTensor_new(state);
  THCTensor *cellOutput_t = THCTensor_(new)(state);
  THCTensor *outputGates_t = THCTensor_(new)(state);
  THCTensor *normalizingConstants_t = THCTensor_(new)(state);

  int inputsSeen = 0;
  for (int t = 0; t < seqLength; t++) {
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
    THCudaIntTensor_narrow(state, batches_t, batches, 0, inputsSeen, numOutArcs_t);

    inputsSeen += numOutArcs_t;

    nThreads = numOutArcs_t * hiddenSize;

    lstmElemwise<real><<<GET_BLOCKS(nThreads), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
        t, hiddenSize, batchSize,
        THCudaIntTensor_data(state, targets_t),
        THCudaIntTensor_data(state, batches_t),
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
    THCTensor_(select)(state, outputGates_t, outputGates, 0, t);
    THCTensor_(select)(state, normalizingConstants_t, normalizingConstants, 0, t);
    THCTensor_(select)(state, output_t, output, 0, t + 1);

    nThreads = batchSize * hiddenSize;

    calculateState<<<GET_BLOCKS(nThreads), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
      hiddenSize, batchSize,
      THCTensor_(data)(state, cellOutput_t),
      THCTensor_(data)(state, outputGates_t),
      THCTensor_(data)(state, normalizingConstants_t),
      THCTensor_(data)(state, output_t)
    );
  }

}

void THNN_(MultiscaleLSTM_backward)(
          THCState *state,
          // Inputs
          THCTensor *input,
          THCTensor *gradInput,
          THCudaIntTensor *targets,
          THCudaIntTensor *batches,
          THCudaIntTensor *origins,
          // Inputs
          THCTensor *output,
          THCTensor *gradOutput,
          THCTensor *cellOutput,
          THCTensor *gradCellOutput,
          // Parameters
          THCTensor *inputWeight,
          THCTensor *gradInputWeight,
          THCTensor *recurrentWeight,
          THCTensor *gradRecurrentWeight,
          THCTensor *bias,
          THCTensor *gradBias,
          // Buffers
          THCudaIntTensor *numOutArcs,
          THCudaIntTensor *numInArcs,
          THCTensor *normalizingConstants,  // Incoming arcs per step and batch
          THCTensor *xW,
          THCTensor *hR,
          THCTensor *gradHR,
          THCTensor *gates,
          THCTensor *gradGates,
          THCTensor *outputGates,
          THCTensor *gradOutputGates,
          int batchSize,
          float scale)
{
  // Get sizes
  int seqLength = THCTensor_(size)(state, output, 0) - 1;
  int totalInputs = THCTensor_(size)(state, input, 0);
  int inputSize = THCTensor_(size)(state, input, 1);
  int hiddenSize = THCTensor_(size)(state, recurrentWeight, 0);

  // Resize buffers
  THCTensor_(resizeAs)(state, gradHR, hR);
  THCTensor_(resizeAs)(state, gradInput, input);
  THCTensor_(resizeAs)(state, gradGates, gates);
  THCTensor_(resizeAs)(state, gradCellOutput, cellOutput);
  THCTensor_(resizeAs)(state, gradOutputGates, outputGates);

  // Accumulation tensors set to zero
  THCTensor_(zero)(state, gradHR);
  THCTensor_(zero)(state, gradGates);
  THCTensor_(zero)(state, gradCellOutput);

  // Sort arcs by destinations instead of origins
  // TODO Re-use buffers instead of allocating each time
  // THCudaIntTensor* targets_ = THCudaIntTensor_newClone(state, targets);
  // THCudaIntTensor* batches_ = THCudaIntTensor_newClone(state, batches);
  // THCudaIntTensor* origins_ = THCudaIntTensor_newClone(state, origins);

  // thrust::device_ptr<int> targets_ptr(THCudaIntTensor_data(state, targets_));
  // thrust::device_ptr<int> batches_ptr(THCudaIntTensor_data(state, batches_));
  // thrust::device_ptr<int> origins_ptr(THCudaIntTensor_data(state, origins_));

  // typedef thrust::tuple<thrust::device_ptr<int>, thrust::device_ptr<int>, thrust::device_ptr<int> > Tuple;
  // typedef thrust::zip_iterator<Tuple> ZipIterator;
  // ZipIterator iter(thrust::make_tuple(targets_ptr, batches_ptr, origins_ptr));

  // thrust::sort_by_key(targets_ptr, targets_ptr + totalInputs, iter);

  // Create tensors to view slices
  THCTensor *gradOutput_t = THCTensor_(new)(state);
  THCTensor *gradGates_t = THCTensor_(new)(state);
  THCTensor *gradCellOutput_t = THCTensor_(new)(state);
  THCTensor *gradOutputGates_t = THCTensor_(new)(state);

  // Create tensors to hold the slices at each step
  THCTensor *output_t = THCTensor_(new)(state);
  THCTensor *hR_t = THCTensor_(new)(state);
  THCTensor *gradHR_t = THCTensor_(new)(state);
  THCTensor *xW_t = THCTensor_(new)(state);
  THCTensor *gates_t = THCTensor_(new)(state);
  THCudaIntTensor *targets_t = THCudaIntTensor_new(state);
  THCudaIntTensor *batches_t = THCudaIntTensor_new(state);
  THCTensor *cellOutput_t = THCTensor_(new)(state);
  THCTensor *outputGates_t = THCTensor_(new)(state);
  THCTensor *normalizingConstants_t = THCTensor_(new)(state);

  int nThreads;

  int inputsSeen = 0;
  for (int t = seqLength - 1; t >= 0; t--) {
    THCTensor_(select)(state, cellOutput_t, cellOutput, 0, t + 1);
    THCTensor_(select)(state, gradCellOutput_t, gradCellOutput, 0, t + 1);
    THCTensor_(select)(state, outputGates_t, outputGates, 0, t);
    THCTensor_(select)(state, gradOutputGates_t, gradOutputGates, 0, t);
    THCTensor_(select)(state, gradOutput_t, gradOutput, 0, t + 1);
    THCTensor_(select)(state, normalizingConstants_t, normalizingConstants, 0, t);

    nThreads = batchSize * hiddenSize;

    calculateGradState<real><<<GET_BLOCKS(nThreads), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
      hiddenSize, batchSize,
      THCTensor_(data)(state, cellOutput_t),
      THCTensor_(data)(state, gradCellOutput_t),
      THCTensor_(data)(state, outputGates_t),
      THCTensor_(data)(state, gradOutputGates_t),
      THCTensor_(data)(state, normalizingConstants_t),
      THCTensor_(data)(state, gradOutput_t)
    );

    int numOutArcs_t = THCudaIntTensor_get1d(state, numOutArcs, t);
    inputsSeen += numOutArcs_t;

    THCTensor_(select)(state, hR_t, hR, 0, t);
    THCTensor_(select)(state, gradHR_t, gradHR, 0, t);
    THCTensor_(narrow)(state, xW_t, xW, 0, totalInputs - inputsSeen, numOutArcs_t);
    THCTensor_(narrow)(state, gates_t, gates, 0, totalInputs - inputsSeen, numOutArcs_t);
    THCTensor_(narrow)(state, gradGates_t, gradGates, 0, totalInputs - inputsSeen, numOutArcs_t);
    THCudaIntTensor_narrow(state, targets_t, targets, 0, totalInputs - inputsSeen, numOutArcs_t);
    THCudaIntTensor_narrow(state, batches_t, batches, 0, totalInputs - inputsSeen, numOutArcs_t);

    nThreads = numOutArcs_t * hiddenSize;

    gradLstmElemwise<real><<<GET_BLOCKS(nThreads), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
        t, hiddenSize, batchSize,
        THCudaIntTensor_data(state, targets_t),
        THCudaIntTensor_data(state, batches_t),
        numOutArcs_t,
        THCTensor_(data)(state, hR_t),
        THCTensor_(data)(state, gradHR_t),
        THCTensor_(data)(state, xW_t),
        THCTensor_(data)(state, bias),
        THCTensor_(data)(state, gates_t),
        THCTensor_(data)(state, gradGates_t),
        THCTensor_(data)(state, cellOutput),
        THCTensor_(data)(state, gradCellOutput),
        THCTensor_(data)(state, outputGates),
        THCTensor_(data)(state, gradOutputGates)
    );

    // TODO Separate streams or batched GEMM
    THCTensor_(select)(state, output_t, output, 0, t);
    #ifdef THC_REAL_IS_FLOAT
    THCudaBlas_Sgemm(
    #elif defined(THC_REAL_IS_HALF)
    THCudaBlas_Hgemm(
    #elif defined(THC_REAL_IS_DOUBLE)
    THCudaBlas_Dgemm(
    #endif
      state,
      'n', 't',
      hiddenSize * 4,
      hiddenSize,
      batchSize,
      ScalarConvert<float, real>::to(scale),
      THCTensor_(data)(state, gradHR_t),
      hiddenSize * 4,
      THCTensor_(data)(state, output_t),
      hiddenSize,
      ScalarConvert<int, real>::to(1),
      THCTensor_(data)(state, gradRecurrentWeight),
      hiddenSize * 4
    );

    THCTensor_(select)(state, gradOutput_t, gradOutput, 0, t);
    #ifdef THC_REAL_IS_FLOAT
    THCudaBlas_Sgemm(
    #elif defined(THC_REAL_IS_HALF)
    THCudaBlas_Hgemm(
    #elif defined(THC_REAL_IS_DOUBLE)
    THCudaBlas_Dgemm(
    #endif
      state,
      't', 'n',
      hiddenSize,
      batchSize,
      hiddenSize * 4,
      ScalarConvert<int, real>::to(1),
      THCTensor_(data)(state, recurrentWeight),
      hiddenSize * 4,
      THCTensor_(data)(state, gradHR_t),
      hiddenSize * 4,
      ScalarConvert<int, real>::to(1),
      THCTensor_(data)(state, gradOutput_t),
      hiddenSize
    );

  }

  #ifdef THC_REAL_IS_FLOAT
  THCudaBlas_Sgemm(
  #elif defined(THC_REAL_IS_HALF)
  THCudaBlas_Hgemm(
  #elif defined(THC_REAL_IS_DOUBLE)
  THCudaBlas_Dgemm(
  #endif
    state,
    'n', 't',
    4 * hiddenSize,
    inputSize,
    totalInputs,
    ScalarConvert<float, real>::to(scale),
    THCTensor_(data)(state, gradGates),
    4 * hiddenSize,
    THCTensor_(data)(state, input),
    inputSize,
    ScalarConvert<int, real>::to(0),
    THCTensor_(data)(state, gradInputWeight),
    4 * hiddenSize
  );

  #ifdef THC_REAL_IS_FLOAT
  THCudaBlas_Sgemm(
  #elif defined(THC_REAL_IS_HALF)
  THCudaBlas_Hgemm(
  #elif defined(THC_REAL_IS_DOUBLE)
  THCudaBlas_Dgemm(
  #endif
    state,
    't', 'n',
    inputSize,
    totalInputs,
    4 * hiddenSize,
    ScalarConvert<int, real>::to(1),
    THCTensor_(data)(state, inputWeight),
    4 * hiddenSize,
    THCTensor_(data)(state, gradGates),
    4 * hiddenSize,
    ScalarConvert<int, real>::to(0),
    THCTensor_(data)(state, gradInput),
    inputSize
  );
}

#endif
