require "fbtorch"
require "cunn"
require "cutorch"
require "nngraph"
local params=torch.reload("./parse")
cutorch.setDevice(params.gpu_index)
local model=torch.reload("./model")
model:Initial(params)
if params.save_score then
    model:GetScore()
else
    model:Distill()
end
