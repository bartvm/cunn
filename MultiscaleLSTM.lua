local MultiscaleLSTM, parent = torch.class('nn.MultiscaleLSTM', 'nn.Module')

-- Initial hidden and cell state are zero

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

  self.numInArcs = torch.IntTensor()
  self.numOutArcs = torch.IntTensor()

  -- Gradients
  self.gradCellOutput = torch.Tensor()
  self._gradOutputGates = torch.Tensor()
  self._gradGates = torch.Tensor()
end

function MultiscaleLSTM:parameters()
  return {self.inputWeight, self.recurrentWeight, self.bias}, {self.gradInputWeight, self.gradRecurrentWeight, self.gradBias}
end

function MultiscaleLSTM:updateOutput(input)
  self.numInArcs = self.numInArcs:cudaInt()
  self.numOutArcs = self.numOutArcs:cudaInt()
  local input, targets, batches, origins = table.unpack(input)
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
    self.numInArcs:cdata(),
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
  local input, targets, batches, origins = table.unpack(input)
  input.THNN.MultiscaleLSTM_updateGradInput(
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
    self.recurrentWeight:cdata(),
    self.bias:cdata(),
    -- Buffers
    self.numInArcs:cdata(),
    self._xW:cdata(),
    self._hR:cdata(),
    self._gates:cdata(),
    self._gradGates:cdata(),
    self._outputGates:cdata(),
    self._gradOutputGates:cdata(),
    -- Config
    self.batchSize
  )
  return {self.gradInput, nil, nil, nil, nil}
end
