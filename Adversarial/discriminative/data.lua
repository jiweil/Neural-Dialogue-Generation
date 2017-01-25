require "torchx"
local stringx = require('pl.stringx')
local Data={}

local function split(str)
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

function Data:Initial(params)
    self.params=params;
end

function Data:get_batch(Source)
    local Words_s={}
    local Padding_s={}
    local Mask_s={}
    local Left_s={}
    for k=1,#Source do
        local max_s=-100;
        for i=1,#Source[k]do
            if Source[k][i]:size(2)>max_s then
                max_s=Source[k][i]:size(2)
            end
        end
        Words_s[k]=torch.Tensor(#Source[k],max_s):fill(1);
        Padding_s[k]=torch.Tensor(#Source[k],max_s):fill(0)
        for i=1,#Source[k] do
            Words_s[k]:sub(i,i,max_s-Source[k][i]:size(2)+1,max_s):copy(Source[k][i]);
            Padding_s[k]:sub(i,i,max_s-Source[k][i]:size(2)+1,max_s):fill(1);
        end
        Mask_s[k]={};
        Left_s[k]={};
        for i=1,Words_s[k]:size(2) do
            Mask_s[k][i]=torch.LongTensor(torch.find(Padding_s[k]:sub(1,-1,i,i),0))
            Left_s[k][i]=torch.LongTensor(torch.find(Padding_s[k]:sub(1,-1,i,i),1))
        end
    end
    return Words_s,Mask_s,Left_s,Padding_s
end

function Data:read_train(open_pos_train_file,open_neg_train_file)
    local Source={};
    local labels=torch.Tensor(self.params.batch_size);
    local End=0;
    for k=1,self.params.dialogue_length do
        Source[k]={};
    end
    local i=0;
    while 1==1 do
        i=i+1;
        local str=open_pos_train_file:read("*line");
        if str==nil then End=1 break end
        strings=stringx.split(str,"|")
        for k=1,#strings do
            Source[k][i]=split(stringx.strip(strings[k]))
        end
        labels[i]=1;
        if i==self.params.batch_size/2 then
            break;
        end
    end
    while 1==1 do
        i=i+1;
        local str=open_neg_train_file:read("*line");
        if str==nil then End=1 break end
        strings=stringx.split(str,"|")
        for k=1,#strings do
            Source[k][i]=split(stringx.strip(strings[k]))
        end
        labels[i]=2;
        if i==self.params.batch_size then
            break;
        end
    end
    if End==1 then
        return End,{},{},{},{}
    end
    local Words_s,Masks_s,Left_t,Padding_s=self:get_batch(Source)
    return End,labels,Words_s,Masks_s,Left_t,Padding_s
end

return Data
