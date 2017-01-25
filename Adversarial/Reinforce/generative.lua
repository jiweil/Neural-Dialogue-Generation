local model={}
local base_model=torch.reload("../../Decode/decode_model")
setmetatable(model,{ __index = base_model })

function model:GenerateSample()
    self.mode="decoding"
    if self.params.sample then 
        self.sample_target=self:sample()
        for i,v in pairs(self.sample_target)do
            self.sample_target[i]=torch.cat(torch.Tensor(1,1):fill(self.Data.EOS):cuda(),torch.cat(v,torch.Tensor(1,1):fill(self.Data.EOT):cuda(),2),2)
        end
    else
        self.params.beam_size=1
        self.params.max_length=20
        self.params.min_length=1
        self.params.MMI=false
        local history=self:decode_BS()
        self.sample_target={}
        for i=1,#history do
            local score=-10000000;
            local string=""
            for j=1,#history[i]do
                if history[i][j].score>score then
                    string=history[i][j].string;
                    score=history[i][j].score;
                end
            end
            self.sample_target[#self.sample_target+1]=torch.cat(torch.Tensor({{self.Data.EOS}}),torch.cat(string,torch.Tensor({self.Data.EOT}))) 
        end
    end
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
    if self.params.dictPath~="" and self.params.dictPath~=nil then
        self:ReadDict()
    end
    self:CollectSampleForIllustration(self.params.dev_file)
end

function model:Integrate(first)
    if first then
        self.Word_t_reserve=self.Word_t
        self.Mask_t_reserve=self.Mask_t
        self.Left_t_reserve=self.Left_t
        self.Padding_t_reserve=self.Padding_t

        self.Word_t=self.Words_sample;
        self.Mask_t=self.Masks_sample
        self.Left_t=self.Left_sample
        self.Padding_t=self.Padding_sample
    else
        self.Word_t=self.Word_t_reserve
        self.Mask_t=self.Mask_t_reserve
        self.Left_t=self.Left_t_reserve
        self.Padding_t=self.Padding_t_reserve
    end
end

function model:PartialSample(staringEmbedding,starting_word,current_time) 
--sampling for partially decoded sequences until the end
    local finish={}
    self.mode="decoding"
    local trans_context=torch.reshape(self.context,self.context:size(1),1,self.context:size(2),self.context:size(3))
    trans_context=torch.repeatTensor(trans_context,1,self.params.MonteCarloExample_N,1,1);
    trans_context=torch.reshape(trans_context,trans_context:size(1)*trans_context:size(2),trans_context:size(3),trans_context:size(4))
    local trans_padding=torch.repeatTensor(self.Padding_s,1,self.params.MonteCarloExample_N);
    trans_padding=torch.reshape(trans_padding,self.params.MonteCarloExample_N*self.Padding_s:size(1),self.Padding_s:size(2))
    for t=1,self.params.max_length-current_time do
        local lstm_input=self:clone_(staringEmbedding);
        table.insert(lstm_input,trans_context);
        table.insert(lstm_input,starting_word);
        table.insert(lstm_input,trans_padding)
        self.lstms_t[1]:evaluate()
        local output=self.lstms_t[1]:forward(lstm_input);
        staringEmbedding={};
        for i=1,2*self.params.layers do
            staringEmbedding[i]=self:copy(output[i]);
        end
        local _,prob=unpack(self.softmax:forward({output[#output],torch.Tensor(output[#output]:size(1)):fill(1):cuda()}))
        prob=nn.Exp():cuda():forward(prob);
        if not self.params.allowUNK then
            prob:sub(1,-1,1,1):fill(0)
        end
        starting_word=torch.multinomial(prob,1);
        starting_word=torch.reshape(starting_word,starting_word:size(1));
        for i=1,starting_word:size(1)do
            if finish[i]==nil then
                if starting_word[i]==self.Data.EOT or (self.partial_history[i]:eq(self.Data.EOT)):sum()~=0 then
                    finish[i]=1 
                else
                    self.partial_history[i]=torch.cat(self.partial_history[i],torch.Tensor(1,1):fill(starting_word[i]):cuda())
                end
            end
        end
        local all_decoded=true;
        for i=1,staringEmbedding[1]:size(1)do
            if finish[i]==nil then
                all_decoded=false;
                break;
            end
        end
        if all_decoded then
            break;
        end
    end
end

function model:ComputeMonteCarloReward(current_time)
    local augment=torch.reshape(torch.repeatTensor(self.Word_t:sub(1,-1,2,current_time+1),1,self.params.MonteCarloExample_N),self.params.MonteCarloExample_N*self.Word_t:size(1),current_time)
    self.partial_history={};
    for i=1,augment:size(1)do
        self.partial_history[i]=augment:sub(i,i,1,-1);
    end
    local starting_word=torch.reshape(torch.repeatTensor(self.Word_t:sub(1,-1,current_time+1,current_time+1),1,self.params.MonteCarloExample_N),self.params.MonteCarloExample_N*self.Word_t:size(1))
    local staringEmbedding={};
    for i=1,2*self.params.layers do
        staringEmbedding[i]=torch.reshape(torch.repeatTensor(self.store_t[current_time][i],1,self.params.MonteCarloExample_N),self.params.MonteCarloExample_N*self.store_t[current_time][i]:size(1),self.store_t[current_time][i]:size(2))
    end
    self:PartialSample(staringEmbedding,starting_word,current_time)
end

function model:model_backward(vanilla)
    local d_source=torch.zeros(self.context:size(1),self.context:size(2),self.context:size(3)):cuda();
    local d_output={};
    for ll=1,self.params.layers do
        table.insert(d_output,torch.zeros(self.Word_s:size(1),self.params.dimension):cuda());
        table.insert(d_output,torch.zeros(self.Word_s:size(1),self.params.dimension):cuda());
    end
    local sum_err=0;
    local total_num=0;
    local End=self.Word_t:size(2)-1
    if not self.params.vanillaReinforce then
        End=self.Word_t:size(2)-2
    end
    for t=self.Word_t:size(2)-1,1,-1 do
        local current_word=self.Word_t:select(2,t+1);
        local softmax_output=self.softmax:forward({self.softmax_h[t],current_word});
        local err=softmax_output[1];
        sum_err=sum_err+err[1]
        total_num=total_num+self.Left_t[t+1]:size(1);
        if self.mode=="train" then
            local d_pred=torch.Tensor(softmax_output[2]:size()):fill(0):cuda()
            local d_error=torch.Tensor({0})
            if not vanilla then
                for i=1,self.Word_t:size(1) do
                    if self.Padding_t[i][t+1]==1 then
                        assert(self.params.vanillaReinforce~=nil)
                        if self.params.vanillaReinforce then
                            d_pred[i][current_word[i]]=self.last_reward[i]
                        else
                            if t==self.Word_t:size(2)-1 then
                                d_pred[i][current_word[i]]=self.MonteCarloReward[i][t-1]
                            else
                                d_pred[i][current_word[i]]=self.MonteCarloReward[i][t]
                            end
                        end
                    end
                end
            else
                d_error=torch.Tensor({1})
            end
            local dh=self.softmax:backward({self.softmax_h[t],current_word},{d_error,d_pred})
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
    --print(1/torch.exp(-sum_err/total_num))
    return sum_err,total_num
end


return model
