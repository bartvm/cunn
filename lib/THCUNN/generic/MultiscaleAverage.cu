#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/MultiscaleAverage.cu"
#else

#include "../common.h"

void THNN_(MultiscaleAverage_updateOutput)(
          THCState *state,
          THCTensor *embeddings,
          THCudaIntTensor *targets,
          THCudaIntTensor *batches,
          THCTensor *output,
          THCTensor *normalizingConstants,  // Incoming arcs per step and batch
          int batchSize)
{
  // Get sizes
  int totalInputs = THCudaIntTensor_size(state, targets, 0);
  int seqLength = THCudaIntTensor_maxall(state, targets);
  int embeddingSize = THCTensor_(size)(state, embeddings, 1);

  THCTensor_(resize2d)(state, normalizingConstants, seqLength, batchSize);
  THCTensor_(resize3d)(state, output, seqLength, batchSize, embeddingSize);
  THCTensor_(zero)(state, output);
  THCTensor_(zero)(state, normalizingConstants);

  int nThreads = totalInputs;

  countArcs<real><<<GET_BLOCKS(nThreads), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
    batchSize, totalInputs, seqLength,
    THCudaIntTensor_data(state, targets),
    THCudaIntTensor_data(state, batches),
    THCTensor_(data)(state, normalizingConstants)
  );

  nThreads = totalInputs * embeddingSize;

  // Count the number of arcs going in and out at each step
  average<real><<<GET_BLOCKS(nThreads), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
    batchSize, totalInputs, seqLength, embeddingSize,
    THCudaIntTensor_data(state, targets),
    THCudaIntTensor_data(state, batches),
    THCTensor_(data)(state, output),
    THCTensor_(data)(state, embeddings),
    THCTensor_(data)(state, normalizingConstants)
  );

}

void THNN_(MultiscaleAverage_updateGradInput)(
          THCState *state,
          // Inputs
          THCTensor *embeddings,
          THCudaIntTensor *targets,
          THCudaIntTensor *batches,
          THCTensor *gradEmbeddings,
          THCTensor *gradOutput,
          THCTensor *normalizingConstants,  // Incoming arcs per step and batch
          int batchSize)
{
  int totalInputs = THCudaIntTensor_size(state, targets, 0);
  int seqLength = THCudaIntTensor_maxall(state, targets);
  int embeddingSize = THCTensor_(size)(state, embeddings, 1);

  THCTensor_(resizeAs)(state, gradEmbeddings, embeddings);
  THCTensor_(zero)(state, gradEmbeddings);

  int nThreads = totalInputs * embeddingSize;

  // Count the number of arcs going in and out at each step
  distribute<real><<<GET_BLOCKS(nThreads), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
    batchSize, totalInputs, seqLength, embeddingSize,
    THCudaIntTensor_data(state, targets),
    THCudaIntTensor_data(state, batches),
    THCTensor_(data)(state, gradOutput),
    THCTensor_(data)(state, gradEmbeddings),
    THCTensor_(data)(state, normalizingConstants)
  );

}
#endif
