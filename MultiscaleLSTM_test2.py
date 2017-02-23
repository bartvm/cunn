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

arcType = 'int64'

W = theano.shared(np.array([0.330683768, 0.655926347, 0.475718617, 0.123243622, 0.633974314, 0.909508169, 0.172109187, 0.573160887, 0.587634325, 0.00642378349, 0.419762284, 0.664884865, 0.995248556, 0.625612378, 0.87671113, 0.437151968, 0.121769592, 0.851266503, 0.232840911, 0.762105942, 0.28621015, 0.409843385, 0.0413793214, 0.464297682], dtype=theano.config.floatX).reshape(inputSize, 4 * hiddenSize), name='W')
R = theano.shared(np.array([0.343660146, 0.763680756, 0.770699918, 0.213859662, 0.449855685, 0.167803735, 0.190994456, 0.510802507, 0.619526386, 0.102920979, 0.58286047, 0.706834376, 0.647269726, 0.833611488, 0.125027329, 0.753009796], dtype=theano.config.floatX).reshape(hiddenSize, 4 * hiddenSize), name='R')
b = theano.shared(np.array([0.715777934, 0.973472238, 0.17162253, 0.66155535, 0.124767587, 0.752138913, 0.725227118, 0.987443984], dtype=theano.config.floatX), name='b')
alpha = theano.shared(np.array([0.104775222145, 0.836851796205, 0.301599609992, 0.893472348341, 0.13091611897, 0.884358023, 0.337557848392, 0.714833051434, 0.40658445812, 0.885005045916, 0.698824328285, 0.696169963234, 0.387402091298, 0.0518384984549, 0.519534290615, 0.105123245401, 0.201343546354, 0.305859359918], dtype=theano.config.floatX), name='alpha')
beta = theano.shared(np.array([0.2, 0.3], dtype=theano.config.floatX), name='beta')

input = tensor.matrix('input')
targets = tensor.vector('targets', dtype=arcType)
batches = tensor.vector('batches', dtype=arcType)
origins = tensor.vector('origins', dtype=arcType)
normalizingConstants = tensor.matrix('normalizingConstants')

input_value = np.array([0.415958315, 0.692421913, 0.605638087, 0.71125555, 0.187767833, 0.777086318, 0.335184276, 0.0847264156, 0.637676537, 0.790640533, 0.252421021, 0.975926518], dtype=theano.config.floatX)
input_value.resize(totalInputs, inputSize)
input.tag.test_value = input_value
targets.tag.test_value = np.array([1, 2, 1, 2], dtype=arcType)
batches.tag.test_value = np.array([0, 0, 1, 0], dtype=arcType)
origins.tag.test_value = np.array([0, 0, 0, 1], dtype=arcType)
normalizingConstants.tag.test_value = np.array([[1, 1], [2, 0]], dtype=theano.config.floatX)

initial_state = tensor.matrix('initial_state')
initial_state.tag.test_value = np.array([[0, 1], [2, 3]], dtype=theano.config.floatX)

cellOutput = tensor.zeros((seqLength + 1, batchSize, hiddenSize))
output = tensor.zeros((seqLength + 1, batchSize, hiddenSize))
outputGates = tensor.zeros((seqLength, batchSize, hiddenSize))
outputGates_ = outputGates
# output = tensor.set_subtensor(output[0], tensor.arange(4).reshape((2, 2)))

def ln(x, axis, i, j):
    y = (x - x.mean(axis=axis, keepdims=True)) / (x.std(axis=axis, keepdims=True) + 1e-5) * alpha[i * hiddenSize:j * hiddenSize]
    if i == 8:
        y = y + beta
    return y

xW = input.dot(W)
xW_ln = ln(xW, 1, 0, 4)
hR = tensor.zeros((seqLength, batchSize, 4 * hiddenSize))
hR = tensor.inc_subtensor(hR[0], output[0].dot(R))
gates = tensor.zeros((totalInputs, 4 * hiddenSize))
gates = tensor.set_subtensor(gates[0], xW_ln[0] + hR[0, 0] + b)
gates = tensor.set_subtensor(gates[1], xW_ln[1] + hR[0, 0] + b)
gates = tensor.set_subtensor(gates[2], xW_ln[2] + hR[0, 1] + b)
gates_ = gates

gates = tensor.set_subtensor(gates[:, hiddenSize:3 * hiddenSize], tensor.nnet.sigmoid(gates[:, hiddenSize:3 * hiddenSize]))
gates = tensor.set_subtensor(gates[:, 3 * hiddenSize:], tensor.tanh(gates[:, 3 * hiddenSize:]))

for i in range(3):
    cellOutput = tensor.inc_subtensor(cellOutput[targets[i], batches[i]], cellOutput[0, batches[i]] * gates[i, 2:4] + gates[i, 6:] * gates[i, 4:6])
    outputGates = tensor.inc_subtensor(outputGates[targets[i] - 1, batches[i]], gates[i, :2])
cellOutput_sigma = cellOutput.std(axis=2)
cellOutput = tensor.set_subtensor(cellOutput[1], ln(cellOutput[1], 1, 8, 9))
output = tensor.set_subtensor(output[1], tensor.tanh(cellOutput[1]) * tensor.nnet.sigmoid(outputGates[0]))
output_ = output
hR = tensor.inc_subtensor(hR[1], output[1].dot(R))
hR_ = hR
hR_ln = tensor.set_subtensor(hR[1], ln(hR[1], 1, 4, 8))

gates = tensor.set_subtensor(gates[3], xW_ln[3] + hR_ln[1, 0] + b)
# gates_ = gates
gates = tensor.set_subtensor(gates[3:, hiddenSize:3 * hiddenSize], tensor.nnet.sigmoid(gates[3:, hiddenSize:3 * hiddenSize]))
gates = tensor.set_subtensor(gates[3:, 3 * hiddenSize:], tensor.tanh(gates[3:, 3 * hiddenSize:]))

for i in range(3, 4):
    cellOutput = tensor.inc_subtensor(cellOutput[targets[i], batches[i]], cellOutput[1, batches[i]] * gates[i, 2:4] + gates[i, 6:] * gates[i, 4:6])
    outputGates = tensor.inc_subtensor(outputGates[targets[i] - 1, batches[i]], gates[i, :2])
cellOutput = tensor.set_subtensor(cellOutput[2, 0], cellOutput[2, 0] / 2)
cellOutput_ = cellOutput
# cellOutput_sigma = cellOutput.std(axis=2)
cellOutput = tensor.set_subtensor(cellOutput[2], ln(cellOutput[2], 1, 8, 9))
cellOutput_ln = cellOutput
outputGates = tensor.set_subtensor(outputGates[1, 0], outputGates[1, 0] / 2)
output = tensor.set_subtensor(output[2, 0], tensor.tanh(cellOutput[2, 0]) * tensor.nnet.sigmoid(outputGates[1, 0]))

print('xW', xW.tag.test_value)
print('xW_ln', xW_ln.tag.test_value)
print('hR', hR.tag.test_value)
print('hR_ln', hR_ln.tag.test_value)
print('gates', gates.tag.test_value)
print('cellOutput', cellOutput.tag.test_value)
print('cellOutput_', cellOutput_.tag.test_value)
print('cellOutput_nobias', (cellOutput[2] - beta).tag.test_value)
print('outputGates', outputGates.tag.test_value)
print('output', output.tag.test_value)
print('hR_sigma', hR.std(axis=2).tag.test_value)
print('cellOutput_sigma', cellOutput_sigma.tag.test_value)

cost = tensor.sum(output)
gradOutput, gradOutputGates, gradCellOutput, gradCellOutput_ln, gradInput, gradGates, gradHR, gradInputWeight, gradRecurrentWeight, gradBias, gradAlpha, gradBeta = tensor.grad(cost, [output_, outputGates_, cellOutput_, cellOutput_ln, input, gates_, hR_, W, R, b, alpha, beta])

gradCellOutput.name = 'gradCellOutput'
gradCellOutput_ln.name = 'gradCellOutput_ln'

print('gradOutput', gradOutput.tag.test_value)
print('gradOutputGates', gradOutputGates.tag.test_value)
print('gradCellOutput', gradCellOutput.tag.test_value)
print('gradCellOutput_ln', gradCellOutput_ln.tag.test_value)
print('gradInput', gradInput.tag.test_value)
print('gradGates', gradGates.tag.test_value)
print('gradHR', gradHR.tag.test_value)
print('gradInputWeight', gradInputWeight.tag.test_value)
print('gradRecurrentWeight', gradRecurrentWeight.tag.test_value)
print('gradBias', gradBias.tag.test_value)
print('gradLnBias', gradBeta.tag.test_value)
print('gradLnGain', gradAlpha.tag.test_value)

# from theano.printing import debugprint as dp
# 
# dp(theano.function([input, targets, batches], [gradCellOutput_ln]))

# print(gradCellOutput.eval({input: input.tag.test_value, targets: targets.tag.test_value, batches: batches.tag.test_value}))

# f = theano.function([input, targets, batches], cost)
# def f_part(input):
#     return f(input, targets.tag.test_value, batches.tag.test_value)
# ng = theano.gradient.numeric_grad(f_part, input.tag.test_value)
# print(ng.gf)
