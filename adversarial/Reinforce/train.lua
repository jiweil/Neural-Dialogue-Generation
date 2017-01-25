require "cunn"
require "cutorch"
require "nngraph"
require "fbtorch"
local params=torch.reload("./parse")
local model=torch.reload("./model")
cutorch.manualSeed(123)
cutorch.setDevice(params.gpu_index)
model:Initial(params)
model:train()
