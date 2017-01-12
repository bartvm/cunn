local cutorch = require 'cutorch'
local MultiscaleLSTM, parent = torch.class('nn.MultiscaleLSTM', 'nn.Module')

function MultiscaleLSTM:__init(batchSize, inputSize, hiddenSize)
  parent.__init(self)
  self.batchSize = batchSize

  -- Parameters
  self.inputWeight = torch.Tensor(inputSize, hiddenSize * 4):zero()
  self.recurrentWeight = torch.Tensor(hiddenSize, hiddenSize * 4):zero()
  self.bias = torch.Tensor(hiddenSize * 4):zero()

  -- Buffers
  self.cellOutput = torch.Tensor()
  self._xW = torch.Tensor()
  self._hR = torch.Tensor()
  self._gates = torch.Tensor()
  self._outputGates = torch.Tensor()

  self.numOutArcs = torch.IntTensor()
  self.normalizingConstants = torch.Tensor()

  -- Gradients
  self.gradCellOutput = torch.Tensor()
  self._gradOutputGates = torch.Tensor()
  self._gradGates = torch.Tensor()
  self._gradHR = torch.Tensor()
  self.gradInputWeight = self.inputWeight:clone()
  self.gradRecurrentWeight = self.recurrentWeight:clone()
  self.gradBias = self.bias:clone()

  -- Set the initial states to zero
  self.output:resize(1, batchSize, hiddenSize):zero()
  self.cellOutput:resize(1, batchSize, hiddenSize):zero()

  -- Create streams
  cutorch.reserveStreams(2)
end

function MultiscaleLSTM:parameters()
  return {self.inputWeight, self.recurrentWeight, self.bias}, {self.gradInputWeight, self.gradRecurrentWeight, self.gradBias}
end

function MultiscaleLSTM:updateOutput(input)
  -- These are cast to float tensors when :cuda() is called on the module
  self.numOutArcs = self.numOutArcs:cudaInt()

  local input, targets, origins, batches = table.unpack(input)

  input.THNN.MultiscaleLSTM_updateOutput(
    -- Inputs
    input:cdata(),
    targets:cdata(),
    batches:cdata(),
    origins:cdata(),
    -- Outputs
    self.output:cdata(),
    self.cellOutput:cdata(),
    -- Parameters
    self.inputWeight:cdata(),
    self.recurrentWeight:cdata(),
    self.bias:cdata(),
    -- Buffers
    self.numOutArcs:cdata(),
    self.normalizingConstants:cdata(),
    self._xW:cdata(),
    self._hR:cdata(),
    self._gates:cdata(),
    self._outputGates:cdata(),
    -- Config
    self.batchSize
   )
  return self.output
end

function MultiscaleLSTM:updateGradInput(input, gradOutput)
  -- TODO Separate accGradParameters and updateGradInput into two kernels
  return self:backward(input, gradOutput, 1)
end

function MultiscaleLSTM:backward(input, gradOutput, scale)
  -- NOTE This module changes the gradient w.r.t. the output in place
  scale = scale or 1
  local input, targets, origins, batches = table.unpack(input)
  input.THNN.MultiscaleLSTM_backward(
    -- Inputs
    input:cdata(),
    self.gradInput:cdata(),
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
    self.bias:cdata(),
    self.gradBias:cdata(),
    -- Buffers
    self.numOutArcs:cdata(),
    self.normalizingConstants:cdata(),
    self._xW:cdata(),
    self._hR:cdata(),
    self._gradHR:cdata(),
    self._gates:cdata(),
    self._gradGates:cdata(),
    self._outputGates:cdata(),
    self._gradOutputGates:cdata(),
    -- Config
    self.batchSize,
    scale
  )
  return {self.gradInput, nil, nil, nil}
end

function MultiscaleLSTM:reset()
  self.output[1]:zero()
  self.cellOutput[1]:zero()
end

function MultiscaleLSTM:carry()
  self.output[1]:copy(self.output[-1])
  self.cellOutput[1]:copy(self.cellOutput[-1])
end
