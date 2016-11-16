local MultiscaleLSTM, parent = torch.class('nn.MultiscaleLSTM', 'nn.Module')

-- Initial hidden and cell state are zero

function MultiscaleLSTM:__init(batchSize, inputSize, hiddenSize)
  parent.__init(self)
  self.batchSize = batchSize
  self.inputWeight = torch.Tensor(inputSize, hiddenSize * 4):zero()
  self.recurrentWeight = torch.Tensor(hiddenSize, hiddenSize * 4):zero()
  self.bias = torch.Tensor(hiddenSize * 4):zero()
  self.cellOutput = torch.Tensor()
  self._xW = torch.Tensor()
  self._hR = torch.Tensor()
  self._gates = torch.Tensor()
  self._outputGates = torch.Tensor()
end

function MultiscaleLSTM:parameters()
  return {self.inputWeight, self.recurrentWeight, self.bias}, {self.gradInputWeight, self.gradRecurrentWeight, self.gradBias}
end

function MultiscaleLSTM:updateOutput(input, targets, batchIndices, numArcs, divisors)
   input.THNN.MultiscaleLSTM_updateOutput(
      input:cdata(),
      targets:cdata(),
      batchIndices:cdata(),
      numArcs:cdata(),
      divisors:cdata(),
      self.output:cdata(),
      self.cellOutput:cdata(),
      self.inputWeight:cdata(),
      self.recurrentWeight:cdata(),
      self.bias:cdata(),
      self._xW:cdata(),
      self._hR:cdata(),
      self._gates:cdata(),
      self._outputGates:cdata(),
      self.batchSize
   )
   return self.output
end

function MultiscaleLSTM:updateGradInput(input, gradOutput)
   input.THNN.MultiscaleLSTM_updateGradInput(
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.output:cdata()
   )
   return self.gradInput
end
