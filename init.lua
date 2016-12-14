require "cutorch"
require "nn"
require "cunn.THCUNN"

require('cunn.test')
require('cunn.DataParallelTable')
require('cunn.MultiscaleLSTM')
require('cunn.MultiscaleCriterion')

nn.Module._flattenTensorBuffer['torch.CudaTensor'] = torch.FloatTensor.new
