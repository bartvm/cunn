#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "THCAtomics.cuh"  // For atomicAdd

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>

template <typename T>
__global__ void countArcs(int batchSize, int totalInputs,
                          int* targets, int* batches, int* origins,
                          int* numOutArcs,
                          T* normalizingConstants) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= totalInputs) return;
  atomicAdd(normalizingConstants + (targets[index] - 1) * batchSize + batches[index], ScalarConvert<int, T>::to(1));
  atomicAdd(numOutArcs + origins[index], 1);
}

__global__ void reverseArcs(int batchSize, int totalInputs, int seqLength,
                            int* targets, int* batches,
                            int* origins,
                            int* offsets, int* arcCount) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= totalInputs) return;
}

template <typename T>
__forceinline__ __device__ T sigmoid(T in) {
  T one = ScalarConvert<float, T>::to(1.0);
  return one / (one + THCNumerics<T>::exp(-in));
}

template <typename T>
__global__ void lstmElemwise(int t, int hiddenSize, int batchSize,
                             int* targets, int* batches, int numOutArcs_t,
                             T* hR, T* xW, T* bias, T* gates,
                             T* cellOutput, T* outputGates) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= numOutArcs_t * hiddenSize) return;

  int input = index / hiddenSize;
  int offset = index % hiddenSize;
  int inputIndex = offset + 4 * input * hiddenSize;
  int hiddenIndex = offset + 4 * batches[input] * hiddenSize;

  T g[4];

  // TODO Add bias to output gate after aggregation
  for (int i = 0; i < 4; i++) {
    g[i] = xW[i * hiddenSize + inputIndex] +
           hR[i * hiddenSize + hiddenIndex] +
           bias[i * hiddenSize + offset];
  }

  T out_gate = g[0];
  T forget_gate = sigmoid<T>(g[1]);
  T in_gate = sigmoid<T>(g[2]);
  T cell_gate = THCNumerics<T>::tanh(g[3]);

  gates[0 * hiddenSize + inputIndex] = out_gate;
  gates[1 * hiddenSize + inputIndex] = forget_gate;
  gates[2 * hiddenSize + inputIndex] = in_gate;
  gates[3 * hiddenSize + inputIndex] = cell_gate;

  // Now distribute cell states and input gates to destinations
  int originIndex = offset + t * batchSize * hiddenSize + batches[input] * hiddenSize;
  int destinationIndex = offset + targets[input] * batchSize * hiddenSize + batches[input] * hiddenSize;
  cellOutput[destinationIndex] += (cellOutput[originIndex] * forget_gate) + (cell_gate * in_gate);
  outputGates[destinationIndex - batchSize * hiddenSize] += out_gate;
}

template <typename T>
__global__ void calculateState(int hiddenSize, int batchSize,
                               T* cellOutput, T* outputGates,
                               T* normalizingConstants, T* output) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= batchSize * hiddenSize) return;

  int batch = index / hiddenSize;
  cellOutput[index] /= normalizingConstants[batch];
  outputGates[index] /= normalizingConstants[batch];
  output[index] = THCNumerics<T>::tanh(cellOutput[index]) * sigmoid<T>(outputGates[index]);
}

template <typename T>
__global__ void calculateGradState(int hiddenSize, int batchSize,
                                   T* cellOutput, T* gradCellOutput,
                                   T* outputGates, T* gradOutputGates,
                                   T* normalizingConstants, T* gradOutput) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= batchSize * hiddenSize) return;

  int batch = index / hiddenSize;
  // NOTE Tanh and sigmoid activations here are recalculated
  T cellActivation = THCNumerics<T>::tanh(cellOutput[index]);
  T outputGateActivation = sigmoid<T>(outputGates[index]);
  gradCellOutput[index] += (1 - cellActivation * cellActivation) * gradOutput[index] * outputGateActivation;
  gradOutputGates[index] = cellActivation * gradOutput[index] * outputGateActivation * (1 - outputGateActivation);
  gradCellOutput[index] /= normalizingConstants[batch];
  gradOutputGates[index] /= normalizingConstants[batch];
}

template <typename T>
__global__ void gradLstmElemwise(int t, int hiddenSize, int batchSize,
                                 int* targets, int* batches, int numOutArcs_t,
                                 T* hR, T* gradHR, T* xW, T* bias,
                                 T* gates, T* gradGates,
                                 T* cellOutput, T* gradCellOutput,
                                 T* outputGates, T* gradOutputGates) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= numOutArcs_t * hiddenSize) return;

  int input = index / hiddenSize;
  int offset = index % hiddenSize;
  int inputIndex = offset + 4 * input * hiddenSize;
  int originIndex = offset + t * batchSize * hiddenSize + batches[input] * hiddenSize;
  int targetIndex = offset + targets[input] * batchSize * hiddenSize + batches[input] * hiddenSize;

  gradCellOutput[originIndex] += gradCellOutput[targetIndex] * gates[1 * hiddenSize + inputIndex];
  gradGates[0 * hiddenSize + inputIndex] += gradOutputGates[targetIndex - batchSize * hiddenSize];
  gradGates[1 * hiddenSize + inputIndex] += gradCellOutput[targetIndex] * cellOutput[originIndex] * gates[1 * hiddenSize + inputIndex] * (1 - gates[1 * hiddenSize + inputIndex]);
  gradGates[2 * hiddenSize + inputIndex] += gradCellOutput[targetIndex] * gates[3 * hiddenSize + inputIndex] * gates[2 * hiddenSize + inputIndex] * (1 - gates[2 * hiddenSize + inputIndex]);
  gradGates[3 * hiddenSize + inputIndex] += gradCellOutput[targetIndex] * gates[2 * hiddenSize + inputIndex] * (1 - gates[3 * hiddenSize + inputIndex] * gates[3 * hiddenSize + inputIndex]);

  for (int i = 0; i < 4; i++) {
    gradHR[i * hiddenSize + offset + batches[input] * hiddenSize * 4] += gradGates[i * hiddenSize + inputIndex];
  }
}

#include "generic/MultiscaleLSTM.cu"
#include "THCGenerateFloatTypes.h"
