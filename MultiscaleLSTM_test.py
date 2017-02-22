import theano
import numpy as np
from theano import tensor

theano.config.compute_test_value = 'warn'
theano.config.floatX = 'float64'

inputSize = 3
hiddenSize = 2
totalInputs = 4
seqLength = 2
batchSize = 2
conditionSize = 2

arcType = 'int64'

W = theano.shared(np.array([0.330683768, 0.655926347, 0.475718617, 0.123243622, 0.633974314, 0.909508169, 0.172109187, 0.573160887, 0.587634325, 0.00642378349, 0.419762284, 0.664884865, 0.995248556, 0.625612378, 0.87671113, 0.437151968, 0.121769592, 0.851266503, 0.232840911, 0.762105942, 0.28621015, 0.409843385, 0.0413793214, 0.464297682], dtype=theano.config.floatX).reshape(inputSize, 4 * hiddenSize), name='W')
R = theano.shared(np.array([0.343660146, 0.763680756, 0.770699918, 0.213859662, 0.449855685, 0.167803735, 0.190994456, 0.510802507, 0.619526386, 0.102920979, 0.58286047, 0.706834376, 0.647269726, 0.833611488, 0.125027329, 0.753009796], dtype=theano.config.floatX).reshape(hiddenSize, 4 * hiddenSize), name='R')
C = theano.shared(np.array([0.443660146, 0.763680756, 0.770699918, 0.213859662, 0.449855685, 0.167803735, 0.190994456, 0.510802507, 0.619526386, 0.102920979, 0.58286047, 0.706834376, 0.647269726, 0.833611488, 0.125027329, 0.753009796], dtype=theano.config.floatX).reshape(conditionSize, 4 * hiddenSize), name='C')
b = theano.shared(np.array([0.715777934, 0.973472238, 0.17162253, 0.66155535, 0.124767587, 0.752138913, 0.725227118, 0.987443984], dtype=theano.config.floatX), name='b')

input = tensor.matrix('input')
condition = tensor.tensor3('condition')
targets = tensor.vector('targets', dtype=arcType)
batches = tensor.vector('batches', dtype=arcType)
origins = tensor.vector('origins', dtype=arcType)
normalizingConstants = tensor.matrix('normalizingConstants')

input_value = np.array([0.415958315, 0.692421913, 0.605638087, 0.71125555, 0.187767833, 0.777086318, 0.335184276, 0.0847264156, 0.637676537, 0.790640533, 0.252421021, 0.975926518], dtype=theano.config.floatX)
input_value.resize(totalInputs, inputSize)
condition_value = np.array([0.115958315, 0.692421913, 0.605638087, 0.71125555, 0.187767833, 0.777086318, 0.335184276, 0.0847264156, 0.637676537, 0.790640533, 0.252421021, 0.975926518], dtype=theano.config.floatX)
condition_value.resize(seqLength, batchSize, conditionSize)
input.tag.test_value = input_value
condition.tag.test_value = condition_value
targets.tag.test_value = np.array([1, 2, 1, 2], dtype=arcType)
batches.tag.test_value = np.array([0, 0, 1, 0], dtype=arcType)
origins.tag.test_value = np.array([0, 0, 0, 1], dtype=arcType)
normalizingConstants.tag.test_value = np.array([[1, 1], [2, 0]], dtype=theano.config.floatX)

initial_state = tensor.matrix('initial_state')
initial_state.tag.test_value = np.array([[0, 1], [2, 3]], dtype=theano.config.floatX)

cellOutput = tensor.zeros((seqLength + 1, batchSize, hiddenSize))
cellOutput_ = cellOutput
output = tensor.zeros((seqLength + 1, batchSize, hiddenSize))
outputGates = tensor.zeros((seqLength, batchSize, hiddenSize))
outputGates_ = outputGates
# output = tensor.set_subtensor(output[0], tensor.arange(4).reshape((2, 2)))

xW = input.dot(W)
hR = tensor.zeros((seqLength, batchSize, 4 * hiddenSize))
hR_ = hR
cW = condition.dot(C)
hR = tensor.inc_subtensor(hR[0], output[0].dot(R))
gates = tensor.zeros((totalInputs, 4 * hiddenSize))
gates = tensor.set_subtensor(gates[0], xW[0] + hR[0, 0] + cW[0, 0] + b)
gates = tensor.set_subtensor(gates[1], xW[1] + hR[0, 0] + cW[0, 0] + b)
gates = tensor.set_subtensor(gates[2], xW[2] + hR[0, 1] + cW[0, 1] + b)
gates_ = gates

gates = tensor.set_subtensor(gates[:, hiddenSize:3 * hiddenSize], tensor.nnet.sigmoid(gates[:, hiddenSize:3 * hiddenSize]))
gates = tensor.set_subtensor(gates[:, 3 * hiddenSize:], tensor.tanh(gates[:, 3 * hiddenSize:]))

for i in range(3):
    cellOutput = tensor.inc_subtensor(cellOutput[targets[i], batches[i]], cellOutput[0, batches[i]] * gates[i, 2:4] + gates[i, 6:] * gates[i, 4:6])
    outputGates = tensor.inc_subtensor(outputGates[targets[i] - 1, batches[i]], gates[i, :2])
output = tensor.set_subtensor(output[1], tensor.tanh(cellOutput[1]) * tensor.nnet.sigmoid(outputGates[0]))
output_ = output
hR = tensor.inc_subtensor(hR[1], output[1].dot(R))

gates = tensor.set_subtensor(gates[3], xW[3] + hR[1, 0] + cW[1, 0] + b)
# gates_ = gates
gates = tensor.set_subtensor(gates[3:, hiddenSize:3 * hiddenSize], tensor.nnet.sigmoid(gates[3:, hiddenSize:3 * hiddenSize]))
gates = tensor.set_subtensor(gates[3:, 3 * hiddenSize:], tensor.tanh(gates[3:, 3 * hiddenSize:]))

for i in range(3, 4):
    cellOutput = tensor.inc_subtensor(cellOutput[targets[i], batches[i]], cellOutput[1, batches[i]] * gates[i, 2:4] + gates[i, 6:] * gates[i, 4:6])
    outputGates = tensor.inc_subtensor(outputGates[targets[i] - 1, batches[i]], gates[i, :2])
cellOutput = tensor.set_subtensor(cellOutput[2, 0], cellOutput[2, 0] / 2)
outputGates = tensor.set_subtensor(outputGates[1, 0], outputGates[1, 0] / 2)
output = tensor.set_subtensor(output[2, 0], tensor.tanh(cellOutput[2, 0]) * tensor.nnet.sigmoid(outputGates[1, 0]))

print('xW', xW.tag.test_value)
print('cW', cW.tag.test_value)
print('gates', gates.tag.test_value)
print('cellOutput', cellOutput.tag.test_value)
print('outputGates', outputGates.tag.test_value)
print('output', output.tag.test_value)

cost = tensor.sum(output)
gradOutput, gradOutputGates, gradCellOutput, gradInput, gradGates, gradHR, gradInputWeight, gradRecurrentWeight, gradBias, gradCondition, gradConditionWeight = tensor.grad(cost, [output_, outputGates_, outputGates_, input, gates_, hR_, W, R, b, condition, C])

print('gradOutput', gradOutput.tag.test_value)
print('gradOutputGates', gradOutputGates.tag.test_value)
print('gradCellOutput', gradCellOutput.tag.test_value)
print('gradInput', gradInput.tag.test_value)
print('gradGates', gradGates.tag.test_value)
print('gradHR', gradHR.tag.test_value)
print('gradInputWeight', gradInputWeight.tag.test_value)
print('gradRecurrentWeight', gradRecurrentWeight.tag.test_value)
print('gradBias', gradBias.tag.test_value)
print('gradCondition', gradCondition.tag.test_value)
print('gradConditionWeight', gradConditionWeight.tag.test_value)

# print(gradCellOutput.eval({input: input.tag.test_value, targets: targets.tag.test_value, batches: batches.tag.test_value}))

# f = theano.function([input, targets, batches], cost)
# def f_part(input):
#     return f(input, targets.tag.test_value, batches.tag.test_value)
# ng = theano.gradient.numeric_grad(f_part, input.tag.test_value)
# print(ng.gf)
