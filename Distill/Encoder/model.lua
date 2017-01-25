local model={}
local base_model=torch.reload("../Decode/decode_model")
setmetatable(model,{ __index = base_model })

function model:Distill()
    self:ComputeTopResponse()
    self:ComputeScore()
end

function model:ComputeScore()
    self.score=torch.Tensor():cuda()
    local open_train_file=io.open(self.params.TrainingData,"r")
    local End=0
    local num=0;
    while End==0 do 
        End,self.Word_s,self.Word_t,self.Mask_s,self.Mask_t,self.Left_s,self.Left_t,self.Padding_s,self.Padding_t=self.Data:read_train(open_train_file)
        self.mode="decoding"
        self.Word_s=self.Word_s:cuda();
        self.Padding_s=self.Padding_s:cuda();
        self:model_forward()
        local embed=torch.Tensor(self.last[2*self.params.layers-1]:size()):cuda():copy(self.last[2*self.params.layers-1]);
        embed=nn.Normalize(2):cuda():forward(embed)
        local score=nn.MM(false,true):cuda():forward({embed,self.TopResponseEmbedding})
        score=torch.max(score,2)
        if self.score:nDimension()==0 then
            self.score=score;
        else
            self.score=torch.cat(self.score,score,1);
        end
        --print(self.score:size())
        num=num+self.params.batch_size
        if num%1280000==0 then
            print(num)
        end
    end
    self.score=torch.reshape(self.score,self.score:size(1));
    local rank_score,index=torch.sort(self.score,true)
    local remove_indexes={}
    for i=1,torch.floor(num/10) do
        remove_indexes[index[i]]=1;
    end
    --print(self.score)
    num=0;
    local open_train=io.open(self.params.TrainingData,"r")
    local output=io.open(self.params.OutputFile,"w")
    local remove=io.open("encode_a.txt","w")
    while true do
        local line=open_train:read("*line")
        if line==nil then break end
        num=num+1
        if remove_indexes[num]==nil then
            output:write(self.score[num].."\n")
            output:write(line.."\n");
        else
            remove:write(self.score[num].."\n")
            remove:write(line.."\n")
        end
    end
    output:close()
    remove:close()
end

function model:ComputeTopResponse()
    local open_train_file=io.open(self.params.TopResponseFile,"r")
    local End=0
    self.TopResponseEmbedding=torch.Tensor():cuda()
    while End==0 do
        End,self.Word_s,self.Word_t,self.Mask_s,self.Mask_t,self.Left_s,self.Left_t,self.Padding_s,self.Padding_t=self.Data:read_train(open_train_file)
        self.mode="decoding"
        self.Word_s=self.Word_s:cuda();
        self.Padding_s=self.Padding_s:cuda();
        self:model_forward()
        local embed=torch.Tensor(self.last[2*self.params.layers-1]:size()):cuda():copy(self.last[2*self.params.layers-1])
        if self.TopResponseEmbedding:nDimension()==0 then
            self.TopResponseEmbedding=embed;
        else
            self.TopResponseEmbedding=torch.cat(self.TopResponseEmbedding,embed,1);
        end
    end
    self.TopResponseEmbedding=nn.Normalize(2):cuda():forward(self.TopResponseEmbedding)
end

return model
