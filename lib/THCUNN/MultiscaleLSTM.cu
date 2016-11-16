#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"

template <typename T>
__forceinline__ __device__ T sigmoid(T in) {
  T one = ScalarConvert<float, T>::to(1.0);
  return one / (one + THCNumerics<T>::exp(-in));
}

template <typename IndexT, typename T>
__global__ void lstmElemwise(int t, int hiddenSize, int batchSize,
                             IndexT* targets, IndexT* batchIndices, int numArcs_t,
                             T* hR, T* xW, T* bias, T* gates,
                             T* cellOutput, T* outputGates) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= numArcs_t * hiddenSize) return;

  int input = index / hiddenSize;
  int offset = index % hiddenSize;
  int inputIndex = offset + 4 * input * hiddenSize;
  int hiddenIndex = offset + 4 * batchIndices[input] * hiddenSize;

  T g[4];

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
  int originIndex = offset + t * batchSize * hiddenSize + batchIndices[input] * hiddenSize;
  int destinationIndex = offset + targets[input] * batchSize * hiddenSize + batchIndices[input] * hiddenSize;
  cellOutput[destinationIndex] += (cellOutput[originIndex] * forget_gate) + (cell_gate * in_gate);
  outputGates[destinationIndex] += out_gate;
}

template <typename T>
__global__ void calculateState(int hiddenSize, int batchSize,
                               T* cellOutput, T* outputGates, T* divisors, T* output) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= batchSize * hiddenSize) return;

  int batch = index / hiddenSize;
  cellOutput[index] /= divisors[batch];
  outputGates[index] /= divisors[batch];
  output[index] = THCNumerics<T>::tanh(cellOutput[index]) * sigmoid<T>(outputGates[index]);
}

#include "generic/MultiscaleLSTM.cu"
#include "THCGenerateFloatTypes.h"
