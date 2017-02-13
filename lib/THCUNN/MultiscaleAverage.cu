#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "THCAtomics.cuh"  // For atomicAdd

template <typename T>
__global__ void countArcs(int batchSize, int totalInputs, int seqLength,
                          int* targets, int* batches,
                          T* normalizingConstants) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= totalInputs || targets[index] == seqLength) return;
  atomicAdd(normalizingConstants + targets[index] * batchSize + batches[index], ScalarConvert<int, T>::to(1));
}

template <typename T>
__global__ void average(int batchSize, int totalInputs, int seqLength, int embeddingSize,
                        int* targets, int* batches,
                        T* output, T* embeddings, T* normalizingConstants) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= totalInputs * embeddingSize) return;
  int input = index / embeddingSize;
  if (targets[input] == seqLength) return;
  int offset = index % embeddingSize;

  int outputIndex = targets[input] * embeddingSize * batchSize + batches[input] * embeddingSize + offset;
  int normalizingIndex = targets[input] * batchSize + batches[input];

  atomicAdd(output + outputIndex, embeddings[index] / normalizingConstants[normalizingIndex]);
}

template <typename T>
__global__ void distribute(int batchSize, int totalInputs, int seqLength, int embeddingSize,
                           int* targets, int* batches,
                           T* gradOutput, T* gradEmbeddings, T* normalizingConstants) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= totalInputs * embeddingSize) return;
  int input = index / embeddingSize;
  if (targets[input] == seqLength) return;
  int offset = index % embeddingSize;

  int outputIndex = targets[input] * embeddingSize * batchSize + batches[input] * embeddingSize + offset;
  int normalizingIndex = targets[input] * batchSize + batches[input];

  gradEmbeddings[index] = gradOutput[outputIndex] / normalizingConstants[normalizingIndex];
}

#include "generic/MultiscaleAverage.cu"
#include "THCGenerateFloatTypes.h"
