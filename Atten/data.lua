require "torchx"
local stringx = require('pl.stringx')

local Data={}

local function reverse(input)
    local length=input:size(2);
    local output=torch.Tensor(1,length);
    for i=1,length do
        output[1][i]=input[1][length-i+1];
    end
    return output;
end

function Data:split(str)
    local split=stringx.split(str," ");
    local tensor=torch.Tensor(1,#split):zero()
    local count=0;
    for i=1,#split do
        if split[i]~=nil and split[i]~="" then
            count=count+1
            tensor[1][count]=tonumber(split[i]);
        end
    end
    return tensor;
end


function Data:get_batch(Sequences,isSource)
    local max_length=-100;
    local Words,Padding,Mask,Left
    for i=1,#Sequences do
        if Sequences[i]:size(2)>max_length then
            max_length=Sequences[i]:size(2);
        end
    end
    if isSource then
        Words=torch.Tensor(#Sequences,max_length):fill(1);
    else
        Words=torch.Tensor(#Sequences,max_length):fill(self.target_dummy);
    end
    Padding=torch.Tensor(#Sequences,max_length):fill(0);
    for i=1,#Sequences do
        if isSource then
            Words:sub(i,i,max_length-Sequences[i]:size(2)+1,max_length):copy(Sequences[i]);
            Padding:sub(i,i,max_length-Sequences[i]:size(2)+1,max_length):fill(1);
        else
            Words:sub(i,i,1,Sequences[i]:size(2)):copy(Sequences[i]);
            Padding:sub(i,i,1,Sequences[i]:size(2)):fill(1);
        end
    end
    Mask={}
    Left={}
    for i=1,Words:size(2) do
        Mask[i]=torch.LongTensor(torch.find(Padding:sub(1,-1,i,i),0))
        Left[i]=torch.LongTensor(torch.find(Padding:sub(1,-1,i,i),1))
    end
    Words=Words:cuda()
    Padding=Padding:cuda()
    return Words,Mask,Left,Padding
end


function Data:Initial(params)
    self.params=params;
    self.EOT=self.params.vocab_target; -- end of target
    self.EOS=self.params.vocab_target-1; --end of source, you can think it as a buffer between source and target
    self.beta=self.params.vocab_target-2;
    self.target_dummy=self.params.vocab_target-3;
end

function Data:read_train(open_train_file)
    local Y={}; 
    local Source={}; 
    local Target={};
    local i=0;
    local End=0;
    while 1==1 do
        i=i+1;
        local str=open_train_file:read("*line");
        if str==nil then
            End=1
            break;
        end
        two_strings=stringx.split(str,"|")
        assert(self.params.reverse~=nil)
        if self.params.reverse then
            Source[i]=reverse(self:split(stringx.strip(two_strings[1])))
        else Source[i]=self:split(stringx.strip(two_strings[1]))
        end
        Target[i]=torch.cat(torch.Tensor({{self.EOS}}),torch.cat(self:split(stringx.strip(two_strings[2])),torch.Tensor({self.EOT})))
        if i==self.params.batch_size then
            break;
        end
    end
    if End==1 then
        return End,{},{},{},{},{},{}
    end
    Words_s,Masks_s,Left_s,Padding_s=self:get_batch(Source,true)
    Words_t,Masks_t,Left_t,Padding_t=self:get_batch(Target,false)
    return End,Words_s,Words_t,Masks_s,Masks_t,Left_s,Left_t,Padding_s,Padding_t,Source,Target
end

return Data
