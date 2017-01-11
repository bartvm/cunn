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
          THCTensor *normalizingConstants,  // Incoming arcs per step and batch
          THCTensor *xW,
          THCTensor *hR,
          THCTensor *gates,
          THCTensor *outputGates,
          // Config
          int batchSize)
{
  // Get sizes
  int totalInputs = THCudaIntTensor_size(state, targets, 0);
  int inputSize = THCTensor_(size)(state, input, 1);
  int hiddenSize = THCTensor_(size)(state, recurrentWeight, 0);

  // The sequence length is the number of hidden states, excluding the initial
  // This means it's equal to the number of elements in the sequence
  thrust::device_ptr<int> targets_ptr(THCudaIntTensor_data(state, targets));
  int seqLength = *thrust::max_element(targets_ptr, targets_ptr + totalInputs);

  // Resize outputs
  // NOTE They are one longer than the sequence to hold the initial state
  THCTensor_(resize3d)(state, output, seqLength + 1, batchSize, hiddenSize);
  THCTensor_(resize3d)(state, cellOutput, seqLength + 1, batchSize, hiddenSize);

  // Resize buffers
  THCTensor_(resize2d)(state, xW, totalInputs, 4 * hiddenSize);
  THCTensor_(resize2d)(state, gates, totalInputs, 4 * hiddenSize);

  THCTensor_(resize3d)(state, hR, seqLength, batchSize, 4 * hiddenSize);
  THCTensor_(resize3d)(state, outputGates, seqLength, batchSize, hiddenSize);

  THCTensor_(resize2d)(state, normalizingConstants, seqLength, batchSize);
  THCudaIntTensor_resize1d(state, numOutArcs, seqLength);

  // Set cellOutput to zero but leave initial state alone
  THCTensor* cellOutput_ = THCTensor_(newNarrow)(state, cellOutput, 0, 1, seqLength);
  THCTensor_(zero)(state, cellOutput_);

  // Accumulation tensors need to be set to 0 too
  THCTensor_(zero)(state, outputGates);
  THCTensor_(zero)(state, normalizingConstants);
  THCudaIntTensor_zero(state, numOutArcs);

  int nThreads = totalInputs;

  // Count the number of arcs going in and out at each step
  countArcs<real><<<GET_BLOCKS(nThreads), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
    batchSize, totalInputs,
    THCudaIntTensor_data(state, targets),
    THCudaIntTensor_data(state, batches),
    THCudaIntTensor_data(state, origins),
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
    // The number of arcs we must process at this time step
    int numOutArcs_t = THCudaIntTensor_get1d(state, numOutArcs, t);

    if (numOutArcs_t != 0) {
      // Transform the hidden state
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
    }

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
    if (numOutArcs_t == 0) {
      continue;
    }
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
    THCState_setCurrentStreamIndex(state, 1);
    cublasSetStream(THCState_getCurrentBlasHandle(state), THCState_getCurrentStream(state));
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

    THCState_setCurrentStreamIndex(state, 2);
    cublasSetStream(THCState_getCurrentBlasHandle(state), THCState_getCurrentStream(state));
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

    THCState_setCurrentStreamIndex(state, 0);

  }

  THCState_setCurrentStreamIndex(state, 1);
  cublasSetStream(THCState_getCurrentBlasHandle(state), THCState_getCurrentStream(state));
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

  THCState_setCurrentStreamIndex(state, 2);
  cublasSetStream(THCState_getCurrentBlasHandle(state), THCState_getCurrentStream(state));
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

  THCState_setCurrentStreamIndex(state, 0);

  THCTensor_(sum)(state, gradBias, gradGates, 0);
  THCTensor_(squeeze1d)(state, gradBias, gradBias, 0);
}

void THNN_(MultiscaleCriterion_updateOutput)(
    // Inputs
    THCState *state,
    THCTensor *input,
    THCudaIntTensor *targets,
    THCudaIntTensor *batches,
    THCudaIntTensor *origins,
    THCudaIntTensor *arcs,
    // Output
    THCTensor *output,
    // Buffers
    THCTensor *stateProbs,
    THCTensor *gradStateProbs,
    THCudaIntTensor *numOutArcs, // Per time step
    THCudaIntTensor *seqLengths,
    bool ignoreLast)
{
  int totalInputs = THCudaIntTensor_size(state, targets, 0);
  int seqLength = THCTensor_(size)(state, input, 0) - (ignoreLast ? 1 : 0);
  int batchSize = THCTensor_(size)(state, input, 1);
  int dictSize = THCTensor_(size)(state, input, 2);

  // Resize buffers and output
  THCudaIntTensor_resize1d(state, seqLengths, batchSize);
  THCTensor_(resize2d)(state, stateProbs, seqLength + 1, batchSize);
  THCudaIntTensor_resize1d(state, numOutArcs, seqLength);
  THCTensor_(resize1d)(state, output, 1);

  // Set accumlating tensors to zero
  THCudaIntTensor_zero(state, numOutArcs);
  THCudaIntTensor_zero(state, seqLengths);
  THCTensor_(zero)(state, output);

  // Initial state probabilities are 1 (so 0 in log-space)
  THCTensor_(fill)(state, stateProbs, THCNumerics<real>::min());
  THCTensor_(fill)(state, THCTensor_(newSelect)(state, stateProbs, 0, 0), ScalarConvert<float, real>::to(0));
  THCudaCheck(cudaDeviceSynchronize());

  // Find the sequence lengths of each example in batch as well as the number of out arcs
  int nThreads = totalInputs;
  findSeqLengths<<<GET_BLOCKS(nThreads), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
    totalInputs,
    THCudaIntTensor_data(state, targets),
    THCudaIntTensor_data(state, batches),
    THCudaIntTensor_data(state, origins),
    THCudaIntTensor_data(state, seqLengths),
    THCudaIntTensor_data(state, numOutArcs)
  );

  THCudaIntTensor *targets_t = THCudaIntTensor_new(state);
  THCudaIntTensor *batches_t = THCudaIntTensor_new(state);
  THCudaIntTensor *origins_t = THCudaIntTensor_new(state);
  THCudaIntTensor *arcs_t = THCudaIntTensor_new(state);
  THCTensor *input_t= THCTensor_(new)(state);

  // Calculate the actual state probabilities
  int inputsSeen = 0;
  for (int t = 0; t < seqLength; t++) {
    int numOutArcs_t = THCudaIntTensor_get1d(state, numOutArcs, t);

    THCudaIntTensor_narrow(state, targets_t, targets, 0, inputsSeen, numOutArcs_t);
    THCudaIntTensor_narrow(state, batches_t, batches, 0, inputsSeen, numOutArcs_t);
    THCudaIntTensor_narrow(state, origins_t, origins, 0, inputsSeen, numOutArcs_t);
    THCudaIntTensor_narrow(state, arcs_t, arcs, 0, inputsSeen, numOutArcs_t);
    THCTensor_(select)(state, input_t, input, 0, t);

    inputsSeen += numOutArcs_t;

    nThreads = numOutArcs_t;
    if (numOutArcs_t != 0) {
      calculateStateProbs<real><<<GET_BLOCKS(nThreads), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
        batchSize, dictSize, numOutArcs_t,
        THCTensor_(data)(state, input_t),
        THCTensor_(data)(state, stateProbs),
        THCudaIntTensor_data(state, targets_t),
        THCudaIntTensor_data(state, batches_t),
        THCudaIntTensor_data(state, origins_t),
        THCudaIntTensor_data(state, arcs_t)
      );
    }
  }

  // We set the gradients to 1 already
  THCTensor_(resizeAs)(state, gradStateProbs, stateProbs);
  THCTensor_(zero)(state, gradStateProbs);

  nThreads = batchSize;
  sumStateProbs<real><<<GET_BLOCKS(nThreads), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
    batchSize,
    THCTensor_(data)(state, stateProbs),
    THCudaIntTensor_data(state, seqLengths),
    THCTensor_(data)(state, output),
    THCTensor_(data)(state, gradStateProbs)
  );

  // The cost can be divided by batchSize, the sum of sequence lengths, or totalInputs
  // THCTensor_(div)(state, output, output, ScalarConvert<float, real>::to(totalInputs));
}

void THNN_(MultiscaleCriterion_updateGradInput)(
    // Inputs
    THCState *state,
    THCTensor *input,
    THCTensor *gradInput,
    THCudaIntTensor *targets,
    THCudaIntTensor *batches,
    THCudaIntTensor *origins,
    THCudaIntTensor *arcs,
    // Output
    THCTensor *output,
    // Buffers
    THCTensor *stateProbs,
    THCTensor *gradStateProbs,
    THCudaIntTensor *numOutArcs, // Per time step
    THCudaIntTensor *seqLengths,
    bool ignoreLast)
{
  int totalInputs = THCudaIntTensor_size(state, targets, 0);
  int seqLength = THCTensor_(size)(state, input, 0) - (ignoreLast ? 1 : 0);
  int batchSize = THCTensor_(size)(state, input, 1);
  int dictSize = THCTensor_(size)(state, input, 2);

  THCTensor_(resizeAs)(state, gradInput, input);
  THCTensor_(zero)(state, gradInput);

  THCudaIntTensor *targets_t = THCudaIntTensor_new(state);
  THCudaIntTensor *batches_t = THCudaIntTensor_new(state);
  THCudaIntTensor *origins_t = THCudaIntTensor_new(state);
  THCudaIntTensor *arcs_t = THCudaIntTensor_new(state);
  THCTensor *input_t= THCTensor_(new)(state);
  THCTensor *gradInput_t= THCTensor_(new)(state);

  int nThreads;

  int inputsSeen = 0;
  for (int t = seqLength - 1; t >= 0; t--) {
    int numOutArcs_t = THCudaIntTensor_get1d(state, numOutArcs, t);
    inputsSeen += numOutArcs_t;

    THCudaIntTensor_narrow(state, targets_t, targets, 0, totalInputs - inputsSeen, numOutArcs_t);
    THCudaIntTensor_narrow(state, batches_t, batches, 0, totalInputs - inputsSeen, numOutArcs_t);
    THCudaIntTensor_narrow(state, origins_t, origins, 0, totalInputs - inputsSeen, numOutArcs_t);
    THCudaIntTensor_narrow(state, arcs_t, arcs, 0, totalInputs - inputsSeen, numOutArcs_t);
    THCTensor_(select)(state, input_t, input, 0, t);
    THCTensor_(select)(state, gradInput_t, gradInput, 0, t);

    nThreads = numOutArcs_t;
    calculateGradStateProbs<real><<<GET_BLOCKS(nThreads), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
      batchSize, dictSize, numOutArcs_t,
      THCTensor_(data)(state, input_t),
      THCTensor_(data)(state, gradInput_t),
      THCTensor_(data)(state, stateProbs),
      THCTensor_(data)(state, gradStateProbs),
      THCudaIntTensor_data(state, targets_t),
      THCudaIntTensor_data(state, batches_t),
      THCudaIntTensor_data(state, origins_t),
      THCudaIntTensor_data(state, arcs_t)
    );

  }

}
#endif
