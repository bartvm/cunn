#include "THCUNN.h"
#include "THCHalfAutoNumerics.cuh"
#include "THCAtomics.cuh"  // For atomicAdd

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

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
__global__ void normalizeState(int hiddenSize, int batchSize,
                               T* cellOutput, T* outputGates,
                               T* normalizingConstants) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= batchSize * hiddenSize) return;

  int batch = index / hiddenSize;
  // States can be unreachable, in case we divide by zero here. In principle
  // not a problem, but the results nans propagate through the linear layer so
  // better to avoid
  if (THCNumerics<T>::ne(normalizingConstants[batch], ScalarConvert<float, T>::to(0))) {
    cellOutput[index] /= normalizingConstants[batch];
    outputGates[index] /= normalizingConstants[batch];
  }
}

template <typename T>
__global__ void calculateState(int hiddenSize, int batchSize,
                               T* cellOutput, T* outputGates,
                               T* output) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= batchSize * hiddenSize) return;

  output[index] = THCNumerics<T>::tanh(cellOutput[index]) * sigmoid<T>(outputGates[index]);
}

template <typename T>
__global__ void calculateGradState(int hiddenSize, int batchSize,
                                   T* cellOutput, T* gradCellOutput,
                                   T* outputGates, T* gradOutputGates,
                                   T* gradOutput) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= batchSize * hiddenSize) return;

  // NOTE Tanh and sigmoid activations here are recalculated
  T cellActivation = THCNumerics<T>::tanh(cellOutput[index]);
  T outputGateActivation = sigmoid<T>(outputGates[index]);
  gradCellOutput[index] += (1 - cellActivation * cellActivation) * gradOutput[index] * outputGateActivation;
  gradOutputGates[index] = cellActivation * gradOutput[index] * outputGateActivation * (1 - outputGateActivation);
}

template <typename T>
__global__ void normalizeGradState(int hiddenSize, int batchSize,
                                   T* gradCellOutput, T* gradOutputGates,
                                   T* normalizingConstants) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= batchSize * hiddenSize) return;

  int batch = index / hiddenSize;
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
    atomicAdd(gradHR + i * hiddenSize + offset + batches[input] * hiddenSize * 4, gradGates[i * hiddenSize + inputIndex]);
  }
}

__global__ void findSeqLengths(int totalInputs, int* targets, int* batches,
                               int* origins, int* seqLengths, int* numOutArcs) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= totalInputs) return;

  atomicMax(seqLengths + batches[index], targets[index]);
  atomicAdd(numOutArcs + origins[index], 1);
}

template <typename T>
__global__ void calculateStateProbs(int batchSize, int dictSize, int numOutArcs_t,
                                    T* input, T* stateProbs,
                                    int* targets, int* batches, int* origins, int* arcs) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= numOutArcs_t) return;

  int targetIndex = targets[index] * batchSize + batches[index];
  int originIndex = origins[index] * batchSize + batches[index];
  int arcIndex = batches[index] * dictSize + (arcs[index] - 1);

  T stateProb = stateProbs[targetIndex];
  T pathProb = input[arcIndex] + stateProbs[originIndex];

  T minProb = pathProb < stateProb ? pathProb : stateProb;
  T maxProb = pathProb > stateProb ? pathProb : stateProb;

  stateProbs[targetIndex] = maxProb + THCNumerics<T>::log1p(THCNumerics<T>::exp(minProb - maxProb));
}

template <typename T>
__global__ void calculateGradStateProbs(int batchSize, int dictSize, int numOutArcs_t,
                                        T* input, T* gradInput,
                                        T* stateProbs, T* gradStateProbs,
                                        int* targets, int* batches, int* origins, int* arcs) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= numOutArcs_t) return;

  int targetIndex = targets[index] * batchSize + batches[index];
  int originIndex = origins[index] * batchSize + batches[index];
  int arcIndex = batches[index] * dictSize + (arcs[index] - 1);

  T stateProb = stateProbs[targetIndex];
  T pathProb = input[arcIndex] + stateProbs[originIndex];
  T gradStateProb = gradStateProbs[targetIndex];

  T arcGrad = THCNumerics<T>::exp(pathProb - stateProb) * gradStateProb;
  gradInput[arcIndex] = arcGrad;
  atomicAdd(gradStateProbs + originIndex, arcGrad);

}

template <typename T>
__global__ void sumStateProbs(int batchSize, T* stateProbs, int* seqLengths, T* output, T* gradStateProbs) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= batchSize) return;

  // Set the initial gradient to -1 (because we are minimizing NLL)
  gradStateProbs[seqLengths[index] * batchSize + index] = ScalarConvert<float, T>::to(-1);
  atomicAdd(output, stateProbs[seqLengths[index] * batchSize + index]);
}

template <typename T>
__global__ void layerNormalization(int batchSize, int dim, T* input, T* output, T* mu, T* sigma, T* gain) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= dim * batchSize) return;
  int example = index / dim;
  int offset = index % dim;
  T eps = ScalarConvert<float, T>::to(1e-5);

  output[index] = (input[index] - mu[example]) / (sigma[example] + eps) * gain[offset];
}

template <typename T>
__global__ void layerNormalizationWithBias(int batchSize, int dim,
                                           T* input, T* output,
                                           T* mu, T* sigma,
                                           T* gain, T* bias) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= dim * batchSize) return;
  int example = index / dim;
  int offset = index % dim;
  T eps = ScalarConvert<float, T>::to(1e-5);

  output[index] = (input[index] - mu[example]) / (sigma[example] + eps) * gain[offset] + bias[offset];
}

template <typename T>
__global__ void gradLayerNormalization(int batchSize, int dim,
                                       T* gradOutput_sum,
                                       T* input, T* gradOutput, T* gradInput,
                                       T* sigma, T* gain) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= dim * batchSize) return;
  int example = index / dim;
  int offset = index % dim;
  T eps = ScalarConvert<float, T>::to(1e-5);

  T before = gradOutput[index];
  gradInput[index] = (gain[offset] * gradOutput[index] - gradOutput_sum[example] - input[index]) / (sigma[example] + eps);
  // printf("%f = (%f * %f - %f) - %f / (%f + %f)\n", gradInput[index], gain[offset], before, gradOutput_sum[example], input[index], sigma[example], eps);
}

#include "generic/MultiscaleLSTM.cu"
#include "THCGenerateFloatTypes.h"
