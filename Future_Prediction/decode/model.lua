local model={}
local base_model=torch.reload("../../Decode/decode_model")

setmetatable(model,{ __index = base_model })

function model:Initial(params)
    local params_file=torch.DiskFile(params.params_file,"r"):binary();
    local model_params=params_file:readObject()
    params_file:close();
    for i,v in pairs(model_params)do 
        if params[i]==nil then
            params[i]=v;
        end
    end
    print(params)
    self.Data:Initial(params)
    self.params=params;
    self.lstm_source =self:lstm_source_()
    self.lstm_target =self:lstm_target_()
    self.softmax =self:softmax_()
    self.Modules={}
    self.Modules[#self.Modules+1]=self.lstm_source
    self.Modules[#self.Modules+1]=self.lstm_target;
    self.Modules[#self.Modules+1]=self.softmax;
    self.lstms_s={}
    self.lstms_t={}
    self.lstms_s[1]=self.lstm_source
    self.lstms_t[1]=self.lstm_target
    self.store_s={}
    self.store_t={}
    self:readModel()
    self.mode="test"
    --self:test()
    self.mode="decoding"
    if self.params.Task=="length" then
        self.FuturePredictModel=torch.reload("../train_length/model")
    else
        self.FuturePredictModel=torch.reload("../train_backward/model")
    end
    params.readSequenceModel=false
    params.readFutureModel=true
    self.FuturePredictModel:Initial(params)
    self.params.dictPath="../../data/movie_25000"
    self:ReadDict()
    if self.params.MMI then
        self:InitialReverse()
    end
end


function model:decode_BS()
    local finish={}
    self:model_forward()
    local span_index1=torch.expand(torch.reshape(torch.expand(torch.reshape(torch.range(1,self.params.beam_size),self.params.beam_size,1),self.params.beam_size,self.params.beam_size),1,self.params.beam_size*self.params.beam_size),self.Word_s:size(1),self.params.beam_size*self.params.beam_size):cuda();
    local span_index2=torch.expand(torch.reshape(self.params.beam_size*(torch.range(1,self.Word_s:size(1))-1),self.Word_s:size(1),1),self.Word_s:size(1),self.params.beam_size):cuda();
    local span_index3=torch.expand(torch.reshape((torch.range(1,self.Word_s:size(1))-1)*self.params.beam_size*self.params.beam_size,self.Word_s:size(1),1),self.Word_s:size(1),self.params.beam_size):cuda();
    local scores
    local completed_history={};
    local batch_max_dec_length;
    if self.params.max_length==0 then batch_max_dec_length=20
    else batch_max_dec_length=self.params.max_length;
    end
    if self.params.target_length~=0 then 
        batch_max_dec_length=self.params.target_length+4
    end
    local beamHistory=torch.ones(self.Word_s:size(1)*self.params.beam_size,batch_max_dec_length):cuda()
    for t=1,batch_max_dec_length do
        local lstm_input=self:clone_(self.last)
        if t==2 then
            local context=torch.reshape(self.context,self.context:size(1),1,self.context:size(2),self.context:size(3))
            context_=torch.expand(context,context:size(1),self.params.beam_size,context:size(3),context:size(4));
            self.context_beam=torch.reshape(context_,context_:size(1)*context_:size(2),context_:size(3),context_:size(4));
            context_=torch.expand(context,context:size(1),self.params.beam_size*self.params.beam_size,context:size(3),context:size(4));
            self.context_beam_future=torch.reshape(context_,context_:size(1)*context_:size(2),context_:size(3),context_:size(4));

            local padding=torch.reshape(self.Padding_s,self.Padding_s:size(1),1,self.Padding_s:size(2));
            local padding_=torch.expand(padding,padding:size(1),self.params.beam_size,padding:size(3));
            self.Padding_s_beam=torch.reshape(padding_,padding_:size(1)*padding_:size(2),padding_:size(3));
            local padding_=torch.expand(padding,padding:size(1),self.params.beam_size*self.params.beam_size,padding:size(3));
            self.Padding_s_beam_future=torch.reshape(padding_,padding_:size(1)*padding_:size(2),padding_:size(3));
        end
        if t==1 then 
            table.insert(lstm_input,self.context);
            table.insert(lstm_input,torch.Tensor(self.Word_s:size(1)):fill(self.Data.EOS):cuda());
            table.insert(lstm_input,self.Padding_s);
        else
            table.insert(lstm_input,self.context_beam);
            table.insert(lstm_input,beamHistory:select(2,t-1));
            table.insert(lstm_input,self.Padding_s_beam);
        end
        self.lstms_t[1]:evaluate()
        local output=self.lstms_t[1]:forward(lstm_input);
        self.last={};
        for i=1,2*self.params.layers do
            self.last[i]=self:copy(output[i]);
        end
        local full_logP=self.softmax:forward({output[#output],torch.Tensor(output[#output]:size(1)):fill(1):cuda()})
        full_logP=full_logP[2]
        if not self.params.allowUNK then
            full_logP:sub(1,-1,1,1):fill(-100)
        end
        local select_logP,select_words=torch.topk(full_logP,self.params.beam_size,2,true,true);
        if t==1 then
            beamHistory:select(2,1):copy(torch.reshape(select_words,select_words:nElement(),1))
            scores=select_logP;
            for i=1,2*self.params.layers do
                local vector=self.last[i];
                vector=torch.reshape(vector,self.Word_s:size(1),1,self.params.dimension);
                vector=torch.expand(vector,self.Word_s:size(1),self.params.beam_size,self.params.dimension);
                self.last[i]=torch.reshape(vector,self.Word_s:size(1)*self.params.beam_size,self.params.dimension);
            end
        else    
            local reshape_score=torch.reshape(scores,scores:nElement(),1);
            local replicate_last={}
            for i=1,2*self.params.layers do
                replicate_last[i]=torch.reshape(self.last[i],self.Word_s:size(1),self.params.beam_size,self.params.dimension)
                replicate_last[i]=torch.repeatTensor(replicate_last[i],1,self.params.beam_size,1)
                replicate_last[i]=torch.reshape(replicate_last[i],self.Word_s:size(1)*self.params.beam_size*self.params.beam_size,self.params.dimension)
            end
            table.insert(replicate_last,self.context_beam_future)
            table.insert(replicate_last,torch.reshape(select_words,self.params.beam_size*self.params.beam_size*self.Word_s:size(1)))
            table.insert(replicate_last,self.Padding_s_beam_future)
            self.lstms_t[1]:evaluate()
            self.lstms_t[1]:forward(replicate_last);
            local output=self.lstms_t[1]:forward(replicate_last);
            output=output[2*self.params.layers-1];
            local future_pred=self.FuturePredictModel.map:forward(output)
            local new_score=torch.expand(reshape_score,reshape_score:size(1),self.params.beam_size)+select_logP;
            local combined_score
            assert(self.params.Task~=nil)
            if self.params.Task=="length" then
                local diff=future_pred-(self.params.target_length-t)
                local PredictError=self.params.PredictorWeight*torch.cmul(diff,diff)
                PredictError=torch.reshape(PredictError,self.Word_s:size(1),self.params.beam_size*self.params.beam_size)
                combined_score=new_score-PredictError
            else
                combined_score=new_score/t+future_pred*self.params.PredictorWeight
            end
            combined_score=torch.reshape(combined_score,self.Word_s:size(1),self.params.beam_size*self.params.beam_size);
            new_score=torch.reshape(new_score,self.Word_s:size(1),self.params.beam_size*self.params.beam_size);
            local reshape_words=torch.reshape(select_words,self.Word_s:size(1),self.params.beam_size*self.params.beam_size);
            local end_boolean_index=reshape_words:eq(self.Data.EOT)
            local extract_scores
            if end_boolean_index:sum()~=0 then
                extract_scores=new_score[end_boolean_index]
                combined_score[end_boolean_index]=-10000;
            end
            if end_boolean_index:sum()~=0 and t>self.params.target_length then
                local end_index=torch.range(1,reshape_words:nElement()):cuda()[end_boolean_index]
                local end_examples=torch.floor((end_index-1)/(self.params.beam_size*self.params.beam_size))+1
                local end_fathers=span_index1[end_boolean_index]
                local end_index_in_history=(end_examples-1)*self.params.beam_size+end_fathers;
                end_index_in_history=end_index_in_history:long()
                local extract_history=beamHistory:index(1,end_index_in_history:cuda());
                for i=1,end_examples:size(1)do
                local example_index=end_examples[i];
                local min_length,max_length
                    if self.params.NBest then
                        local his_sub=extract_history:sub(i,i,1,t-1);
                        if his_sub:ge(self.Data.EOS):sum()==0 then
                            if completed_history[example_index]==nil then
                                completed_history[example_index]={};
                            end
                            local num_instance=#completed_history[example_index]+1
                            completed_history[example_index][num_instance]={}
                            completed_history[example_index][num_instance].string=his_sub
                            completed_history[example_index][num_instance].score=extract_scores[i]/(completed_history[example_index][num_instance].string:size(2)+1);
                        end
                    else
                        if self.params.target_length==0 or completed_history[example_index]==nil then
                            local his_sub=extract_history:sub(i,i,1,t-1);
                            if his_sub:ge(self.Data.EOS):sum()==0 then
                                if completed_history[example_index]==nil then
                                    completed_history[example_index]={};
                                end
                                local num_instance=#completed_history[example_index]+1
                                completed_history[example_index][num_instance]={}
                                completed_history[example_index][num_instance].string=his_sub
                                completed_history[example_index][num_instance].score=extract_scores[i]/(completed_history[example_index][num_instance].string:size(2)+1);
                            end
                        end
                    end
                end
            end
            local select_scores,select_index=torch.topk(combined_score,self.params.beam_size,2,true,true);
            select_index=select_index:double():cuda()
            scores=torch.Tensor(self.Word_s:size(1),self.params.beam_size):fill(0):cuda()
            for k1=1,self.Word_s:size(1) do
                for k2=1,self.params.beam_size do
                    scores[k1][k2]=new_score[k1][select_index[k1][k2]]
                end
            end
            scores=torch.reshape(scores,scores:nElement(),1);
            local select_fathers=(torch.floor((select_index-1)/self.params.beam_size)+1);
            local select_index_in_history=(span_index2+select_fathers)
            select_index_in_history=torch.reshape(select_index_in_history,select_index_in_history:nElement()):long()
            beamHistory=beamHistory:index(1,select_index_in_history);
            local new_added_words_index=torch.reshape(span_index3+select_index,select_index:nElement()):long();
            local new_added_words=torch.reshape(reshape_words,reshape_words:nElement()):index(1,new_added_words_index);
            beamHistory:select(2,t):copy(new_added_words);
            for i=1,2*self.params.layers do
                self.last[i]=self:copy(self.last[i]:index(1,select_index_in_history));
            end
        end
    end
    for i=1,self.Word_s:size(1) do
        if completed_history[i]==nil then
            completed_history[i]={}
            completed_history[i][1]={};
            completed_history[i][1].string=beamHistory:sub((i-1)*self.params.beam_size+1,(i-1)*self.params.beam_size+1,1,-1);
            completed_history[i][1].score=scores[(i-1)*self.params.beam_size+1][1]
        end
    end
    if self.params.MMI then
        completed_history=self:MMI(completed_history)
    end
    return completed_history;
end
return model
