local cutorch = require 'cutorch'
local ConditionedMultiscaleLSTM, parent = torch.class('nn.ConditionedMultiscaleLSTM', 'nn.Module')

function ConditionedMultiscaleLSTM:__init(batchSize, inputSize, conditionSize, hiddenSize)
  parent.__init(self)
  self.batchSize = batchSize

  -- Parameters
  self.inputWeight = torch.Tensor(inputSize, hiddenSize * 4):zero()
  self.conditionWeight = torch.Tensor(conditionSize, hiddenSize * 4):zero()
  self.recurrentWeight = torch.Tensor(hiddenSize, hiddenSize * 4):zero()
  self.bias = torch.Tensor(hiddenSize * 4):zero()

  -- Buffers
  self.cellOutput = torch.Tensor()
  self._xW = torch.Tensor()
  self._hR = torch.Tensor()
  self._cW = torch.Tensor()
  self._gates = torch.Tensor()
  self._outputGates = torch.Tensor()

  self.numOutArcs = torch.IntTensor()
  self.normalizingConstants = torch.Tensor()

  -- Gradients
  self.gradCellOutput = torch.Tensor()
  self.gradCondition = torch.Tensor()
  self._gradOutputGates = torch.Tensor()
  self._gradGates = torch.Tensor()
  self._gradHR = torch.Tensor()
  self.gradInputWeight = self.inputWeight:clone()
  self.gradRecurrentWeight = self.recurrentWeight:clone()
  self.gradConditionWeight = self.conditionWeight:clone()
  self.gradBias = self.bias:clone()

  -- Set the initial states to zero
  self.output:resize(1, batchSize, hiddenSize):zero()
  self.cellOutput:resize(1, batchSize, hiddenSize):zero()

  -- Create streams
  cutorch.reserveStreams(2)

  -- Initialize weights
  self:reset()
end

function ConditionedMultiscaleLSTM:parameters()
  return {self.inputWeight, self.recurrentWeight, self.conditionWeight, self.bias}, {self.gradInputWeight, self.gradRecurrentWeight, self.gradConditionWeight, self.gradBias}
end

function ConditionedMultiscaleLSTM:updateOutput(input)
  -- These are cast to float tensors when :cuda() is called on the module
  self.numOutArcs = self.numOutArcs:cudaInt()

  local input, condition, targets, origins, batches = table.unpack(input)

  input.THNN.ConditionedMultiscaleLSTM_updateOutput(
    -- Inputs
    input:cdata(),
    condition:cdata(),
    targets:cdata(),
    batches:cdata(),
    origins:cdata(),
    -- Outputs
    self.output:cdata(),
    self.cellOutput:cdata(),
    -- Parameters
    self.inputWeight:cdata(),
    self.recurrentWeight:cdata(),
    self.conditionWeight:cdata(),
    self.bias:cdata(),
    -- Buffers
    self.numOutArcs:cdata(),
    self.normalizingConstants:cdata(),
    self._xW:cdata(),
    self._cW:cdata(),
    self._hR:cdata(),
    self._gates:cdata(),
    self._outputGates:cdata(),
    -- Config
    self.batchSize
   )
  return self.output
end

function ConditionedMultiscaleLSTM:updateGradInput(input, gradOutput)
  -- TODO Separate accGradParameters and updateGradInput into two kernels
  return self:backward(input, gradOutput, 1)
end

function ConditionedMultiscaleLSTM:backward(input, gradOutput, scale)
  -- NOTE This module changes the gradient w.r.t. the output in place
  scale = scale or 1
  local input, condition, targets, origins, batches = table.unpack(input)
  input.THNN.ConditionedMultiscaleLSTM_backward(
    -- Inputs
    input:cdata(),
    self.gradInput:cdata(),
    condition:cdata(),
    self.gradCondition:cdata(),
    targets:cdata(),
    batches:cdata(),
    origins:cdata(),
    -- Outputs
    self.output:cdata(),
    gradOutput:cdata(),
    self.cellOutput:cdata(),
    self.gradCellOutput:cdata(),
    -- Parameters
    self.inputWeight:cdata(),
    self.gradInputWeight:cdata(),
    self.recurrentWeight:cdata(),
    self.gradRecurrentWeight:cdata(),
    self.conditionWeight:cdata(),
    self.gradConditionWeight:cdata(),
    self.bias:cdata(),
    self.gradBias:cdata(),
    -- Buffers
    self.numOutArcs:cdata(),
    self.normalizingConstants:cdata(),
    self._xW:cdata(),
    self._hR:cdata(),
    self._cW:cdata(),
    self._gradHR:cdata(),
    self._gates:cdata(),
    self._gradGates:cdata(),
    self._outputGates:cdata(),
    self._gradOutputGates:cdata(),
    -- Config
    self.batchSize,
    scale
  )
  return {self.gradInput, self.gradCondition, nil, nil, nil}
end

function ConditionedMultiscaleLSTM:forget()
  self.output[1]:zero()
  self.cellOutput[1]:zero()
end

function ConditionedMultiscaleLSTM:carry()
  self.output[1]:copy(self.output[-1])
  self.cellOutput[1]:copy(self.cellOutput[-1])
end

function ConditionedMultiscaleLSTM:reset()
  local inputSize = self.inputWeight:size(1)
  local hiddenSize = self.inputWeight:size(2) / 4
  -- Inputweight and bias is Caffe-style Xavier-Glorot
  local stdv = math.sqrt(6 / (inputSize + hiddenSize))
  self.inputWeight:uniform(-stdv, stdv)
  self.bias:zero()
  -- Weights/biases are in order out, forget, in, cell
  -- We set the forget gate to a high value
  self.bias[{{hiddenSize + 1, 2 * hiddenSize}}]:fill(1)

  -- Recurrent weights are orthogonal
  for i = 0, 3 do
    local randMat = torch.Tensor(hiddenSize, hiddenSize):normal(0, 1)
    local U = torch.svd(randMat, 'S')
    self.recurrentWeight[{{}, {i * hiddenSize + 1, (i + 1) * hiddenSize}}]:copy(U)
  end
end
