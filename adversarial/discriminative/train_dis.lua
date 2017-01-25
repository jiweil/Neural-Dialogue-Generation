require "fbtorch"
require "cunn"
require "cutorch"
require "nngraph"
local params=torch.reload("./dis_parse")
local model=torch.reload("./dis_model");

cutorch.manualSeed(123)
cutorch.setDevice(params.gpu_index)
model:Initial(params)
model:train()
