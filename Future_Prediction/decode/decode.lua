require "fbtorch"
require "cunn"
require "cutorch"
require "nngraph"
local params=torch.reload("./parse")
cutorch.setDevice(params.gpu_index)
local decode_model=torch.reload("./model")
decode_model:Initial(params)
decode_model.mode="test"
decode_model:decode()
