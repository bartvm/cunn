local cutorch = require 'cutorch'
local MultiscaleCriterion, parent = torch.class('nn.MultiscaleCriterion', 'nn.Criterion')

function MultiscaleCriterion:__init(ignoreLast)
  self.ignoreLast = (ignoreLast == nil) and true or ignoreLast
  parent.__init(self)

  self._stateProbs = torch.Tensor()
  self._gradStateProbs = torch.Tensor()
  self._output = torch.Tensor()
  self.numOutArcs = torch.IntTensor()
  self.seqLengths = torch.IntTensor()
end

function MultiscaleCriterion:updateOutput(input, target)
  local arcs, targets, origins, batches = table.unpack(target)
  self.numOutArcs = self.numOutArcs:cudaInt()
  self.seqLengths = self.seqLengths:cudaInt()

  input.THNN.MultiscaleCriterion_updateOutput(
    input:cdata(),
    targets:cdata(),
    batches:cdata(),
    origins:cdata(),
    arcs:cdata(),
    self._output:cdata(),
    self._stateProbs:cdata(),
    self._gradStateProbs:cdata(),
    self.numOutArcs:cdata(),
    self.seqLengths:cdata(),
    self.ignoreLast
  )
  self.output = -self._output[1]
  return self.output
end

function MultiscaleCriterion:updateGradInput(input, target)
  local arcs, targets, origins, batches = table.unpack(target)
  input.THNN.MultiscaleCriterion_updateGradInput(
    input:cdata(),
    self.gradInput:cdata(),
    targets:cdata(),
    batches:cdata(),
    origins:cdata(),
    arcs:cdata(),
    self._output:cdata(),
    self._stateProbs:cdata(),
    self._gradStateProbs:cdata(),
    self.numOutArcs:cdata(),
    self.seqLengths:cdata(),
    self.ignoreLast
  )
  return self.gradInput
end
