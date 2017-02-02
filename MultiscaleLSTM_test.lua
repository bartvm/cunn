require 'cunn'
torch.manualSeed(0)

batchSize = 2
inputSize = 3
hiddenSize = 2
dictSize = 4
conditionSize = 2

-- Module
m = nn.MultiscaleLSTM(batchSize, inputSize, conditionSize, hiddenSize)

-- Parameters
m.inputWeight:copy(torch.Tensor{0.330683768, 0.655926347, 0.475718617, 0.123243622, 0.633974314, 0.909508169, 0.172109187, 0.573160887, 0.587634325, 0.00642378349, 0.419762284, 0.664884865, 0.995248556, 0.625612378, 0.87671113, 0.437151968, 0.121769592, 0.851266503, 0.232840911, 0.762105942, 0.28621015, 0.409843385, 0.0413793214, 0.464297682})
m.recurrentWeight:copy(torch.Tensor{0.343660146, 0.763680756, 0.770699918, 0.213859662, 0.449855685, 0.167803735, 0.190994456, 0.510802507, 0.619526386, 0.102920979, 0.58286047, 0.706834376, 0.647269726, 0.833611488, 0.125027329, 0.753009796})
m.conditionWeight:copy(torch.Tensor{0.443660146, 0.763680756, 0.770699918, 0.213859662, 0.449855685, 0.167803735, 0.190994456, 0.510802507, 0.619526386, 0.102920979, 0.58286047, 0.706834376, 0.647269726, 0.833611488, 0.125027329, 0.753009796})
m.bias:copy(torch.Tensor{0.715777934, 0.973472238, 0.17162253, 0.66155535, 0.124767587, 0.752138913, 0.725227118, 0.987443984})

m = m:cuda()

-- Inputs (0-indexed!)
input = torch.CudaTensor{0.415958315, 0.692421913, 0.605638087, 0.71125555, 0.187767833, 0.777086318, 0.335184276, 0.0847264156, 0.637676537, 0.790640533, 0.252421021, 0.975926518}:resize(4, 3)
condition = torch.CudaTensor{0.115958315, 0.692421913, 0.605638087, 0.71125555, 0.187767833, 0.777086318, 0.335184276, 0.0847264156, 0.637676537, 0.790640533, 0.252421021, 0.975926518}:resize(2, 2, 2)
targets = torch.CudaIntTensor{1, 2, 1, 2}
batches = torch.CudaIntTensor{0, 0, 1, 0}
origins = torch.CudaIntTensor{0, 0, 0, 1}
arcs = torch.CudaIntTensor{2, 3, 1, 0}

seqLength = torch.max(targets)

-- Run forward prop
m:updateOutput({input, condition, targets, origins, batches})

-- Run backward prop
gradOutput = torch.ones(seqLength + 1, batchSize, hiddenSize):cuda()
m:backward({input, condition, targets, origins, batches}, gradOutput)

-- Print results and buffers for inspection
-- print('inputWeight', m.inputWeight)
-- print('recurrentWeight', m.recurrentWeight)
-- print('bias', m.bias)
print('output', m.output)
print('cW', m._cW)
print('gates', m._gates)
print('xW', m._xW)
print('hR', m._hR)
print('cellOutput', m.cellOutput)
print('gradCellOutput', m.gradCellOutput)
print('gradOutput', gradOutput)
print('normalizingConstants', m.normalizingConstants)
-- print('outputGates', m._outputGates)
print('gradOutputGates', m._gradOutputGates)
print('gradGates', m._gradGates)
print('gradBias', m.gradBias)
print('gradHR', m._gradHR)
print('gradInput', m.gradInput)
print('gradCondition', m.gradCondition)
print('gradInputWeight', m.gradInputWeight)
print('gradRecurrentWeight', m.gradRecurrentWeight)
print('gradConditionWeight', m.gradConditionWeight)
-- print('numOutArcs', m.numOutArcs)
-- print('normalizingConstants', m.normalizingConstants)

-- Criterion stuff

probs = nn.Sequential()
  :add(nn.Linear(hiddenSize, dictSize))
  :add(nn.LogSoftMax())
probs:cuda()

logprobs = probs:forward(m.output:resize((seqLength + 1) * batchSize, hiddenSize)):resize(seqLength + 1, batchSize, dictSize)
print('probs', torch.exp(logprobs))

m2 = nn.MultiscaleCriterion():cuda()
cost = m2:forward(logprobs, {arcs, targets, origins, batches})
m2:backward(logprobs, {arcs, targets, origins, batches})
print('numOutArs', m2.numOutArcs)
print('seqLengths', m2.seqLengths)
print('stateProbs', m2._stateProbs)
print('gradStateProbs', m2._gradStateProbs)
print('gradInput', m2.gradInput)
print(cost)
