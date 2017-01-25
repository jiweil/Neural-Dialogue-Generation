require "cutorch"
require "nn"
require 'cunn'
require "nngraph"
local stringx = require('pl.stringx')
cutorch.manualSeed(123)
local data=torch.reload("../../Atten/data")

local model={}

function model:Initial(params)
    self.params=params
    local file=torch.DiskFile(self.params.WordMatrix,"r"):binary();
    local embedding=file:readObject();
    self.LookUpTable=nn.LookupTable(embedding:size()):cuda();
    local parameter,_=self.LookUpTable:parameters()
    parameter[1]:copy(embedding:cuda())

    self.getMatrix=nn.Sequential()
    self.getMatrix:add(self.LookUpTable);
    self.getMatrix:add(nn.Sum(2))
    self.getMatrix:add(nn.Normalize(2))
    self.getMatrix=self.getMatrix:cuda()
    self:LoadGram()

    self.top_response_lines=self:ReadFile(self.params.TopResponseFile)
    self.top_response_embedding=self:lines2Embedding(self.top_response_lines)
    if self.params.loadscore then
        local file=torch.DiskFile(self.params.save_score_file,"r"):binary();
        self.all_scores=file:readObject();
        self.all_scores=self.all_scores:double()
    end
end

function model:lines2Embedding(lines)
    local max_length=-100;
    local All_tensors={}
    for i,str in pairs(lines)do
        local split=stringx.split(str," ");
        if #split>max_length then
            max_length=#split
        end
        local tensor=torch.Tensor(1,#split):zero()
        for j=1,#split do
            tensor[1][j]=tonumber(split[j]);
        end
        All_tensors[#All_tensors+1]=tensor;
    end
    local matrix=torch.Tensor(#lines,max_length):fill(1);
    for i,tensor in pairs(All_tensors)do
        matrix:sub(i,i,1,tensor:size(2)):copy(tensor);
    end
    local vector=self.getMatrix:forward(matrix)
    return torch.Tensor(vector:size()):copy(vector):cuda();
end

function model:LoadGram()
    local open_=io.open(self.params.TopResponseFile,"r");
    self.FourGram={}
    while true do
        local line=open_:read("*line");
        if line==nil then break end
        local t=line:find("|");
        line=line:sub(t+1,-1);
        local G=stringx.split(line," ")
        --for i=1,#G-3 do
        if #G>=4 then
            for i=1,1 do
                local gram=G[i].." "..G[i+1].." "..G[i+2].." "..G[i+3];
                if self.FourGram[gram]==nil then
                    self.FourGram[gram]=1
                end
            end
        end
    end 
end

function model:ReadFile(file)
    local open_=io.open(file,"r");
    local lines={}
    while true do
        local line=open_:read("*line");
        if line==nil then break end
        local t=line:find("|");
        lines[#lines+1]=line:sub(t+1,-1);
    end 
    return lines
end

function model:GetScore()
    open_train=io.open(self.params.TrainingData)
    local current_lines={}
    local all_lines={}
    self.all_scores=torch.Tensor():cuda()
    num=0
    while true do
        local line=open_train:read("*line")
        if line==nil then break end
        local splits=stringx.split(line,"|")
        local str=stringx.strip(splits[2])
        current_lines[#current_lines+1]=str
        num=num+1
        if #current_lines%10000000==0 then
            print(num)
            local current_matrix=self:lines2Embedding(current_lines);
            local score=nn.MM(false,true):cuda():forward({current_matrix,self.top_response_embedding})
            score=torch.max(score,2)
            if self.all_scores:nDimension()==0 then
                self.all_scores=score;
            else
                self.all_scores=torch.cat(self.all_scores,score,1);
            end
            current_lines={}
        end
        --[[
        if num==100000 then
            break;
        end
        --]]
    end
    self.all_scores=torch.reshape(self.all_scores,self.all_scores:size(1))
    if self.params.save_score then
        local file=torch.DiskFile(self.params.save_score_file,"w"):binary();
        file:writeObject(self.all_scores);
        file:close()
    end
end

function model:Distill()
    local output=io.open(self.params.OutputFile,"w")
    local reserve=io.open("Glove_reserve_index.txt","w")
    local remove=io.open("Glove_remove_index.txt","w")
    local rank_score,index=torch.topk(self.all_scores,torch.floor(0.3*self.all_scores:size(1)),true)
    local remove_indexes={}
    for i=1,torch.floor(self.params.total_lines*self.params.distill_rate) do
        remove_indexes[index[i]]=1;
    end
    num=0;
    open_train=io.open(self.params.TrainingData)
    local four_distill_num=0
    local cos_distill_num=0
    while true do
        local line=open_train:read("*line")
        if line==nil then break end
        num=num+1
        if num>self.all_scores:size(1) then
            break
        end
        local distill=false
        if remove_indexes[num]~=nil then
            cos_distill_num=cos_distill_num+1
            distill=true;
        end
        local t=line:find("|")
        local target=line:sub(t+1,-1)
        if self.params.distill_four_gram then
            if not distill then
                local G=stringx.split(target," ")
                for i=1,#G-3 do
                    local gram=G[i].." "..G[i+1].." "..G[i+2].." "..G[i+3];
                    if self.FourGram[gram]~=nil then
                        distill=true;
                        four_distill_num=four_distill_num+1
                        break
                    end
                end
            end
        end
        if not distill then
            output:write(line.."\n");
            reserve:write(self.all_scores[num].."\n");
            t=line:find("|")
            reserve:write(target.."\n")
        else
            t=line:find("|")
            remove:write(self.all_scores[num].."\n")
            remove:write(target.."\n")
        end
        --[[
        if num==100000 then
            break;
        end
        --]]
    end
    print(cos_distill_num)
    print(four_distill_num)
end

return model
