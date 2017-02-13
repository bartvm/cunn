local cutorch = require 'cutorch'
local MultiscaleAverage, parent = torch.class('nn.MultiscaleAverage', 'nn.Module')

function MultiscaleAverage:__init(batchSize)
  parent.__init(self)
  self.batchSize = batchSize

  self.normalizingConstants = torch.Tensor()
end

function MultiscaleAverage:updateOutput(input)
  -- These are cast to float tensors when :cuda() is called on the module
  local embeddings, targets, batches = table.unpack(input)

  embeddings.THNN.MultiscaleAverage_updateOutput(
    -- Inputs
    embeddings:cdata(),
    targets:cdata(),
    batches:cdata(),
    -- Outputs
    self.output:cdata(),
    self.normalizingConstants:cdata(),
    -- Config
    self.batchSize
   )
  return self.output
end

function MultiscaleAverage:updateGradInput(input, gradOutput)
  local embeddings, targets, batches = table.unpack(input)

  embeddings.THNN.MultiscaleAverage_updateGradInput(
    -- Inputs
    embeddings:cdata(),
    targets:cdata(),
    batches:cdata(),
    self.gradInput:cdata(),
    -- Outputs
    gradOutput:cdata(),
    self.normalizingConstants:cdata(),
    -- Config
    self.batchSize
   )
  return {self.gradInput, nil, nil}
end
