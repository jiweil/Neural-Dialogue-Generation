require "fbtorch"
require "cunn"
require "cutorch"
require "nngraph"
local params=require("fbcode.experimental.deeplearning.jiwei.babi.rlbabi2..jiwei.distill.Glove.parse")
cutorch.setDevice(params.gpu_index)
local model=require("fbcode.experimental.deeplearning.jiwei.babi.rlbabi2..jiwei.distill.Glove.model")
model:Initial(params)
if params.save_score then
    model:GetScore()
else
    model:Distill()
end
