local model={}
local base_model=torch.reload("../../../Atten/data")
setmetatable(model,{ __index = base_model })

function model:GenerateSample()
    self.mode="decoding"
    self.sample_target=self:sample()

    self.Words_sample,self.Masks_sample,self.Left_sample,self.Padding_sample=self.Data:get_batch(self.sample_target,false)
    self.Words_sample=self.Words_sample:cuda()
    self.Padding_sample=self.Padding_sample:cuda()
end

function model:Initial(params)
    self.Data:Initial(params)
    self.params=params;
    self.lstm_source =self:lstm_source_()
    self.lstm_target =self:lstm_target_()
    self.softmax =self:softmax_()
    self.Modules={}
    self.Modules[#self.Modules+1]=self.lstm_source
    self.Modules[#self.Modules+1]=self.lstm_target;
    self.Modules[#self.Modules+1]=self.softmax;
    self.lstms_s=self:g_cloneManyTimes(self.lstm_source,self.params.source_max_length)
    self.lstms_t=self:g_cloneManyTimes(self.lstm_target,self.params.target_max_length)
    self.store_s={}
    self.store_t={}
    self:readModel()
    self.mode="test"
    --self:test()
    self.mode="decoding"
end

function model:Integrate()
    self.Word_t=self.Words_sample;
    self.Mask_t=self.Masks_sample
    self.Left_t=self.Left_sample
    self.Padding_t=self.Padding_sample
--[[
    self.Word_s=torch.cat(self.Word_s,self.Word_s,1)
    local max_word_t=math.max(self.Word_t:size(2),self.Words_sample:size(2))
    Word_t=torch.Tensor(self.Word_t:size(1)*2,max_word_t)
    Word_t:sub(1,self.Word_t:size(1),1,self.Word_t:size(2)):copy(self.Word_t)
    Word_t:sub(self.Word_t:size(1)+1,self.Word_t:size(1)+self.Words_sample:size(1),1,self.Words_sample:size(2)):copy(self.Words_sample);
    self.Word_t=Word_t;
    self.Padding_s=torch.cat(self.Padding_s,self.Padding_s,1)

    local self.Padding_t=torch.Tensor(Word_t:size()):fill(0):cuda()
    for i=1,#self.target do
        self.Padding_t:sub(i,i,1,self.target:size(2)):fill(1);
    end
    for i=1,#self.sample_target do
        self.Padding_t:sub(i+#self.target,i+#self.target,1,self.sample_target:size(2)):fill(1)
    end
    self.Mask_s={}
    self.Mask_t={}
    self.Left_s={}
    self.Left_t={}
    for i=1,self.Padding_s:size(2)do
        self.Mask_s[i]=torch.LongTensor(torch.find(self.Padding_s:sub(1,-1,i,i),0))
        self.Left_s[i]=torch.LongTensor(torch.find(self.Padding_s:sub(1,-1,i,i),1))
    end
    for i=1,self.Padding_t:size(2)do
        self.Mask_t[i]=torch.LongTensor(torch.find(self.Padding_t:sub(1,-1,i,i),0))
        self.Left_t[i]=torch.LongTensor(torch.find(self.Padding_t:sub(1,-1,i,i),1))
    end
--]]
end

function model:model_backward(batch_n)
    local d_source=torch.zeros(self.context:size(1),self.context:size(2),self.context:size(3)):cuda();
    local d_output={};
    for ll=1,self.params.layers do
        table.insert(d_output,torch.zeros(self.Word_s:size(1),self.params.dimension):cuda());
        table.insert(d_output,torch.zeros(self.Word_s:size(1),self.params.dimension):cuda());
    end
    local sum_err=0;
    local total_num=0;
    for t=self.Word_t:size(2)-1,1,-1 do
        local current_word=self.Word_t:select(2,t+1);
        local softmax_output=self.softmax:forward({self.softmax_h[t],current_word});
        local err=softmax_output[1];
        sum_err=sum_err+err[1]
        total_num=total_num+self.Left_t[t+1]:size(1);
        if self.mode=="train" then
            local d_pred=torch.Tensor(softmax_output[2]:size()):fill(0):cuda()
            for i=1,self.Word_t:size(1) do
                if i>d_pred:size(1) or current_word[i]>d_pred:size(2) or i>self.reward:size(1) then
                    print(i,current_word[i])
                    print(d_pred:size())
                    print(self.reward:size())
                end
                if batch_n==28 then
                    print("i")
                    print(i)
                    print(self.reward)
                    print(current_word)
                end
                if self.Padding_t[i][t+1]==1 then
                    d_pred[i][current_word[i]]=self.reward[i]
                end
            end
            local dh=self.softmax:backward({self.softmax_h[t],current_word},{torch.Tensor({0}),d_pred})
            d_store_t=self:clone_(d_output);
            table.insert(d_store_t,dh[1])        
            local now_input={};
            if t~=1 then    
                now_input=self:clone_(self.store_t[t-1]);
            else
                now_input=self:clone_(self.store_s[self.Word_s:size(2)]);
            end
            table.insert(now_input,self.context);
            table.insert(now_input,self.Word_t:select(2,t));
            table.insert(now_input,self.Padding_s);
            local now_d_input=self.lstms_t[t]:backward(now_input,d_store_t);
            if self.Mask_t[t]:nDimension()~=0 then
                for i=1,2*self.params.layers+2 do
                    now_d_input[i]:indexCopy(1,self.Mask_t[t],torch.zeros(self.Mask_t[t]:size(1),self.params.dimension):cuda())
                end
            end
            d_output={};
            for i=1,2*self.params.layers do
                d_output[i]=self:copy(now_d_input[i])
            end
            d_source:add(now_d_input[2*self.params.layers+1]);
        end
        if self.mode=="train" then
            for t=self.Word_s:size(2),1,-1 do
                local now_input={}
                if t~=1 then
                    now_input=self:clone_(self.store_s[t-1])
                else 
                    for ll=1,self.params.layers do
                        table.insert(now_input,torch.zeros(self.Word_s:size(1),self.params.dimension):cuda());
                        table.insert(now_input,torch.zeros(self.Word_s:size(1),self.params.dimension):cuda());
                    end
                end
                table.insert(now_input,self.Word_s:select(2,t));
                d_output[2*self.params.layers-1]:add(d_source[{{},t,{}}])
                local d_now_output=self.lstms_s[t]:backward(now_input,d_output);
                if self.Mask_s[t]:nDimension()~=0 then
                    for i=1,#d_now_output-1 do
                        d_now_output[i]:indexCopy(1,self.Mask_s[t],torch.zeros(self.Mask_s[t]:size(1),self.params.dimension):cuda())
                    end
                end
                d_output={}
                for i=1,2*self.params.layers do
                    d_output[i]=self:copy(d_now_output[i])
                end
            end
        end
    end
    return sum_err,total_num
end


return model
