#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/MultiscaleLSTM.cu"
#else

#include "../common.h"

void THNN_(MultiscaleLSTM_updateOutput)(
          THCState *state,
          // Inputs
          THCTensor *input,
          THCIndexTensor *targets,
          THCIndexTensor *batchIndices,
          THCIndexTensor *numArcs, // numInputs
          THCTensor *divisors, // TODO This should be calculated instead of provided
          // Outputs
          THCTensor *output,
          THCTensor *cellOutput,
          // Parameters
          THCTensor *inputWeight,
          THCTensor *recurrentWeight,
          THCTensor *bias,
          // Buffers
          THCTensor *xW,
          THCTensor *hR,
          THCTensor *gates,
          THCTensor *outputGates,
          int batchSize)
{
  // TODO Assert everything on same GPU
  // TODO Check input sizes

  int seqLength = THCIndexTensor_(size)(state, numArcs, 0) + 1;
  int totalInputs = THCTensor_(size)(state, input, 0);
  int inputSize = THCTensor_(size)(state, input, 1);
  int hiddenSize = THCTensor_(size)(state, recurrentWeight, 0);

  THCTensor_(resize3d)(state, output, seqLength, batchSize, hiddenSize);
  THCTensor_(resize3d)(state, cellOutput, seqLength, batchSize, hiddenSize);
  THCTensor_(resize2d)(state, xW, totalInputs, 4 * hiddenSize);
  THCTensor_(resize3d)(state, hR, seqLength - 1, batchSize, 4 * hiddenSize);
  THCTensor_(resize2d)(state, gates, totalInputs, 4 * hiddenSize);
  // Can technically be seqLength - 1
  THCTensor_(resize3d)(state, outputGates, seqLength, batchSize, hiddenSize);

  // Initial states are zero
  THCTensor_(zero)(state, output);
  THCTensor_(zero)(state, cellOutput);

  // Accumulation tensors need to be set to 0 too
  THCTensor_(zero)(state, outputGates);

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
  THCIndexTensor *targets_t = THCIndexTensor_(new)(state);
  THCIndexTensor *batchIndices_t = THCIndexTensor_(new)(state);
  THCTensor *cellOutput_t = THCTensor_(new)(state);
  THCTensor *outputGates_t = THCTensor_(new)(state);
  THCTensor *divisors_t = THCTensor_(new)(state);

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

    int numArcs_t = THCIndexTensor_(get1d)(state, numArcs, t);
    THCTensor_(narrow)(state, xW_t, xW, 0, inputsSeen, numArcs_t);
    THCTensor_(narrow)(state, gates_t, gates, 0, inputsSeen, numArcs_t);
    THCIndexTensor_(narrow)(state, targets_t, targets, 0, inputsSeen, numArcs_t);
    THCIndexTensor_(narrow)(state, batchIndices_t, batchIndices, 0, inputsSeen, numArcs_t);

    inputsSeen += numArcs_t;

    int nThreads = numArcs_t * hiddenSize;

    lstmElemwise<long, real><<<GET_BLOCKS(nThreads), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
        t, hiddenSize, batchSize,
        THCIndexTensor_(data)(state, targets_t),
        THCIndexTensor_(data)(state, batchIndices_t),
        numArcs_t,
        THCTensor_(data)(state, hR_t),
        THCTensor_(data)(state, xW_t),
        THCTensor_(data)(state, bias),
        THCTensor_(data)(state, gates_t),
        THCTensor_(data)(state, cellOutput),
        THCTensor_(data)(state, outputGates)
    );

    THCTensor_(select)(state, cellOutput_t, cellOutput, 0, t + 1);
    THCTensor_(select)(state, outputGates_t, outputGates, 0, t + 1);
    THCTensor_(select)(state, divisors_t, divisors, 0, t);
    THCTensor_(select)(state, output_t, output, 0, t + 1);

    nThreads = batchSize * hiddenSize;

    calculateState<<<GET_BLOCKS(nThreads), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
      hiddenSize, batchSize,
      THCTensor_(data)(state, cellOutput_t),
      THCTensor_(data)(state, outputGates_t),
      THCTensor_(data)(state, divisors_t),
      THCTensor_(data)(state, output_t)
    );
  }

}

#endif
