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

  int totalInputs = THCTensor_(size)(state, input, 0);
  int inputSize = THCTensor_(size)(state, input, 1);
  int hiddenSize = THCTensor_(size)(state, recurrentWeight, 0);
  int seqLength = THCIndexTensor_(size)(state, numArcs, 0) + 1;

  THCTensor_(resize3d)(state, output, seqLength, batchSize, hiddenSize);
  THCTensor_(zero)(state, output);
  THCTensor_(resize3d)(state, cellOutput, seqLength, batchSize, hiddenSize);
  THCTensor_(zero)(state, output);
  THCTensor_(resize2d)(state, xW, totalInputs, 4 * hiddenSize);
  THCTensor_(resize3d)(state, hR, seqLength - 1, batchSize, 4 * hiddenSize);
  THCTensor_(resize2d)(state, gates, totalInputs, 4 * hiddenSize);
  THCTensor_(resize3d)(state, outputGates, seqLength, batchSize, hiddenSize);

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

  int inputsSeen = 0;
  for (int t = 0; t < seqLength - 1; t++) {
    // Transform the previous hidden state [batch x hidden -> batch x hidden * 4]
    // output = (R.T * h.T).T = h * R
    THCTensor *output_t = THCTensor_(new)(state);
    THCTensor *hR_t = THCTensor_(new)(state);
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
  }

}

#endif
